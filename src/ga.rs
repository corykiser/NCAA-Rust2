// Genetic Algorithm module for NCAA Bracket Optimization
// Implements proper population-based GA with smart mutation and best-ball scoring

use crate::bracket::{Bracket, FastBracket, ScoreTable, ScoringConfig};
use crate::config::{Config, GaSettings};
use crate::exact::TeamLock;
use crate::ingest::{RcTeam, TournamentInfo};
use crate::tree::{NO_GAME, NUM_ROUNDS, PARENT};
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;

/// Pre-generated pool of random brackets for scoring
/// These represent possible tournament outcomes
/// Uses FastBracket internally for high-performance scoring in the hot path
#[derive(Clone)]
pub struct MonteCarloScenarios {
    /// Full bracket objects (kept for debugging/inspection)
    pub brackets: Vec<Bracket>,
    /// FastBracket versions for high-performance scoring (u8 winner indices)
    pub fast_brackets: Vec<FastBracket>,
    pub size: usize,
    /// Pre-computed score lookup table for fast scoring
    pub score_table: ScoreTable,
}

impl MonteCarloScenarios {
    /// Generate a new simulation pool with random brackets
    pub fn new(tournament: &TournamentInfo, size: usize, scoring_config: &ScoringConfig) -> Self {
        println!("Generating simulation pool of {} brackets...", size);

        let brackets: Vec<Bracket> = (0..size)
            .into_par_iter()
            .map(|_| Bracket::new(tournament, Some(scoring_config)))
            .collect();

        // Convert to FastBrackets for high-performance scoring
        let fast_brackets: Vec<FastBracket> = brackets
            .par_iter()
            .map(|b| FastBracket::from_bracket(b))
            .collect();

        // Pre-compute score lookup table
        let score_table = ScoreTable::new(scoring_config);

        println!("Simulation pool generated.");

        MonteCarloScenarios { brackets, fast_brackets, size, score_table }
    }

    /// Score a single bracket against all simulations using FastBracket
    /// Returns the average score
    pub fn score_bracket(&self, bracket: &Bracket, _scoring_config: &ScoringConfig) -> f64 {
        let table = &self.score_table;
        // Convert input bracket to FastBracket once
        let fast_bracket = FastBracket::from_bracket(bracket);

        let total: f64 = self.fast_brackets
            .par_iter()
            .map(|sim| fast_bracket.score_against(sim, table))
            .sum();

        total / self.size as f64
    }

    /// Score a portfolio using best-ball metric with FastBracket
    /// For each simulation, take the max score among all portfolio brackets
    /// Return the average of these max scores
    pub fn score_portfolio_best_ball(
        &self,
        portfolio: &[Bracket],
        _scoring_config: &ScoringConfig,
    ) -> f64 {
        if portfolio.is_empty() {
            return 0.0;
        }

        let table = &self.score_table;

        // Convert portfolio to FastBrackets once
        let fast_portfolio: Vec<FastBracket> = portfolio
            .iter()
            .map(|b| FastBracket::from_bracket(b))
            .collect();

        let total: f64 = self.fast_brackets
            .par_iter()
            .map(|sim| {
                fast_portfolio
                    .iter()
                    .map(|b| b.score_against(sim, table))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0)
            })
            .sum();

        total / self.size as f64
    }

    /// Score a bracket's marginal contribution to an existing portfolio using FastBracket
    /// This is the increase in best-ball score when adding this bracket
    pub fn combined_best_ball(
        &self,
        bracket: &Bracket,
        existing_portfolio: &[Bracket],
        _scoring_config: &ScoringConfig,
    ) -> f64 {
        if existing_portfolio.is_empty() {
            return self.score_bracket(bracket, &ScoringConfig::default());
        }

        let table = &self.score_table;

        // Convert to FastBrackets once
        let fast_bracket = FastBracket::from_bracket(bracket);
        let fast_portfolio: Vec<FastBracket> = existing_portfolio
            .iter()
            .map(|b| FastBracket::from_bracket(b))
            .collect();

        let total: f64 = self.fast_brackets
            .par_iter()
            .map(|sim| {
                let existing_max = fast_portfolio
                    .iter()
                    .map(|b| b.score_against(sim, table))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                let new_score = fast_bracket.score_against(sim, table);

                // Marginal contribution is how much better we do with this bracket
                new_score.max(existing_max)
            })
            .sum();

        total / self.size as f64
    }
}

/// Directed mutation: pick a team, pick a round, and make that team reach it.
///
/// The previous implementation decided each bit from the moving team's seed
/// alone (`team.seed <= 8`), which ignores who the team is actually playing.
/// The encoding bit means "the lower seed of *this matchup* advances", so a
/// 6-seed needs a different bit against an 11-seed than against a 3-seed —
/// roughly half of all "forced" advances did the opposite of what was intended,
/// and the championship case was decided from one region index when it depends
/// on both finalists. This walks the team's actual path instead, so the bit is
/// always computed against the real opponent.
pub struct TeamRoundMutator;

impl TeamRoundMutator {
    /// Pick a random team and force it to win a random number of games.
    /// Earlier rounds are weighted more heavily because they move more picks.
    pub fn mutate(
        bracket: &Bracket,
        tournament: &TournamentInfo,
        scoring_config: &ScoringConfig,
    ) -> Bracket {
        let mut rng = rand::thread_rng();

        let team_idx = rng.gen_range(0..tournament.teams.len());
        let team = &tournament.teams[team_idx];

        let r: f64 = rng.gen();
        let wins: usize = if r < 0.40 {
            1
        } else if r < 0.65 {
            2
        } else if r < 0.80 {
            3
        } else if r < 0.90 {
            4
        } else if r < 0.97 {
            5
        } else {
            6
        };

        let new_binary = Self::force_team_to_round(&bracket.binary, tournament, team, wins);
        Bracket::new_from_binary(tournament, &new_binary, Some(scoring_config))
    }

    /// Rewrite `binary` so that `team` wins its first `wins` games.
    pub fn force_team_to_round(
        binary: &[bool],
        tournament: &TournamentInfo,
        team: &RcTeam,
        wins: usize,
    ) -> Vec<bool> {
        Self::force_index_to_round(binary, tournament, team.team_index, wins)
    }

    /// As `force_team_to_round`, addressing the team by index.
    ///
    /// Walks up the single path from the team's round-1 game, setting each bit
    /// against the opponent that the already-rewritten bits below actually
    /// deliver. Games above `wins` keep their bits and are re-decoded by the
    /// caller, so the result is always a legal bracket.
    pub fn force_index_to_round(
        binary: &[bool],
        tournament: &TournamentInfo,
        team_index: u8,
        wins: usize,
    ) -> Vec<bool> {
        let mut new_binary = binary.to_vec();
        let mut winners = tournament.decode_winners(&new_binary);

        let mut game = tournament.r1_game_of_team[team_index as usize];

        for _ in 0..wins.min(NUM_ROUNDS) {
            let (a, b) = tournament.participants(game, &winners);
            let opponent = if a == team_index {
                b
            } else if b == team_index {
                a
            } else {
                debug_assert!(
                    false,
                    "team {} is not a participant of game {}",
                    team_index, game
                );
                break;
            };

            new_binary[game] = tournament.winner_bit(team_index, opponent);
            winners[game] = team_index;

            game = PARENT[game];
            if game == NO_GAME {
                break;
            }
        }

        new_binary
    }
}

/// Team-round requirements re-applied to every candidate.
///
/// Mutation and crossover are free to move any pick, so a lock that is only
/// applied to the starting bracket is gone after one generation. Repairing
/// each candidate keeps the constraint true of everything the optimizer ever
/// scores.
#[derive(Debug, Clone, Default)]
pub struct LockSet {
    pub locks: Vec<TeamLock>,
}

impl LockSet {
    pub fn new(locks: Vec<TeamLock>) -> Self {
        LockSet { locks }
    }

    pub fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }

    /// Force every lock back into a bit vector.
    pub fn repair_binary(&self, binary: Vec<bool>, tournament: &TournamentInfo) -> Vec<bool> {
        let mut binary = binary;
        for lock in &self.locks {
            binary = TeamRoundMutator::force_index_to_round(
                &binary,
                tournament,
                lock.team_index,
                lock.wins_required,
            );
        }
        binary
    }

    /// Force every lock back into a bracket, rebuilding only if something moved.
    pub fn repair(
        &self,
        bracket: Bracket,
        tournament: &TournamentInfo,
        scoring_config: &ScoringConfig,
    ) -> Bracket {
        if self.locks.is_empty() || self.is_satisfied(&bracket) {
            return bracket;
        }
        let repaired = self.repair_binary(bracket.binary.clone(), tournament);
        Bracket::new_from_binary(tournament, &repaired, Some(scoring_config))
    }

    /// Whether every lock already holds in `bracket`.
    pub fn is_satisfied(&self, bracket: &Bracket) -> bool {
        self.locks
            .iter()
            .all(|lock| bracket.wins_for(lock.team_index) >= lock.wins_required)
    }
}

/// Individual in the GA population
#[derive(Clone)]
pub struct Individual {
    pub bracket: Bracket,
    pub fitness: f64,
}

impl Individual {
    pub fn new(bracket: Bracket) -> Self {
        Individual {
            bracket,
            fitness: 0.0,
        }
    }

    pub fn with_fitness(bracket: Bracket, fitness: f64) -> Self {
        Individual { bracket, fitness }
    }
}

/// Genetic Algorithm for bracket optimization
pub struct GeneticAlgorithm {
    pub population: Vec<Individual>,
    pub settings: GaSettings,
    pub scoring_config: ScoringConfig,
    pub generation: usize,
    pub best_fitness: f64,
    pub best_bracket: Option<Bracket>,
    /// Re-applied to every candidate; see `LockSet`.
    pub locks: LockSet,
}

impl GeneticAlgorithm {
    /// Create a new GA with random initial population
    pub fn new(
        tournament: &TournamentInfo,
        settings: GaSettings,
        scoring_config: ScoringConfig,
    ) -> Self {
        let population: Vec<Individual> = (0..settings.population_size)
            .into_par_iter()
            .map(|_| Individual::new(Bracket::new(tournament, Some(&scoring_config))))
            .collect();

        GeneticAlgorithm {
            population,
            settings,
            scoring_config,
            generation: 0,
            best_fitness: 0.0,
            best_bracket: None,
            locks: LockSet::default(),
        }
    }

    /// Constrain the search: every candidate, starting with the initial
    /// population, is repaired to satisfy these locks.
    pub fn with_locks(mut self, locks: LockSet, tournament: &TournamentInfo) -> Self {
        if !locks.is_empty() {
            let scoring = self.scoring_config;
            self.population = self
                .population
                .drain(..)
                .map(|ind| Individual::new(locks.repair(ind.bracket, tournament, &scoring)))
                .collect();
        }
        self.locks = locks;
        self
    }

    /// Evaluate fitness for all individuals using simulation pool
    pub fn evaluate_fitness(&mut self, pool: &MonteCarloScenarios) {
        // Parallel fitness evaluation
        let fitnesses: Vec<f64> = self.population
            .par_iter()
            .map(|ind| pool.score_bracket(&ind.bracket, &self.scoring_config))
            .collect();

        // Update fitness values
        for (ind, fitness) in self.population.iter_mut().zip(fitnesses.into_iter()) {
            ind.fitness = fitness;
        }

        // Track best
        if let Some(best) = self.population.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap()
        }) {
            if best.fitness > self.best_fitness {
                self.best_fitness = best.fitness;
                self.best_bracket = Some(best.bracket.clone());
            }
        }
    }

    /// Evaluate fitness for portfolio mode (marginal contribution)
    pub fn evaluate_fitness_portfolio(
        &mut self,
        pool: &MonteCarloScenarios,
        existing_portfolio: &[Bracket],
    ) {
        let fitnesses: Vec<f64> = self.population
            .par_iter()
            .map(|ind| {
                pool.combined_best_ball(&ind.bracket, existing_portfolio, &self.scoring_config)
            })
            .collect();

        for (ind, fitness) in self.population.iter_mut().zip(fitnesses.into_iter()) {
            ind.fitness = fitness;
        }

        if let Some(best) = self.population.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap()
        }) {
            if best.fitness > self.best_fitness {
                self.best_fitness = best.fitness;
                self.best_bracket = Some(best.bracket.clone());
            }
        }
    }

    /// Tournament selection - pick best from random subset
    fn tournament_select(&self, rng: &mut impl Rng) -> &Individual {
        let mut best: Option<&Individual> = None;

        for _ in 0..self.settings.tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            let candidate = &self.population[idx];

            if best.is_none() || candidate.fitness > best.unwrap().fitness {
                best = Some(candidate);
            }
        }

        best.unwrap()
    }

    /// Team-Round crossover - the genes are Team-Round pairs
    /// 1. Start with one parent as the base
    /// 2. Pick N Team-Round pairs that EXIST in the donor bracket
    /// 3. Apply those Team-Round pairs to the child (force those teams to reach those rounds)
    fn crossover(parent1: &Bracket, parent2: &Bracket, tournament: &TournamentInfo, scoring_config: &ScoringConfig, rng: &mut impl Rng) -> Bracket {
        // Pick which parent is the base (50/50)
        let (base, donor) = if rng.gen::<bool>() {
            (parent1, parent2)
        } else {
            (parent2, parent1)
        };

        // Start with base parent's binary
        let mut child_binary = base.binary.clone();

        // Pick N Team-Round pairs from donor to inject (N = 1 to 4)
        let num_injections = rng.gen_range(1..=4);

        for _ in 0..num_injections {
            // Pick a random round (1-6, but later rounds have fewer teams)
            let round: usize = rng.gen_range(1..=6);

            // Find a team that actually reached this round in the donor bracket
            // by looking at the game winners
            if let Some(team) = Self::get_team_at_round(donor, round, rng) {
                // Apply this Team-Round pair to the child
                child_binary = TeamRoundMutator::force_team_to_round(
                    &child_binary,
                    tournament,
                    &team,
                    round,
                );
            }
        }

        Bracket::new_from_binary(tournament, &child_binary, Some(scoring_config))
    }

    /// Get a random team that reached a specific round in the bracket
    fn get_team_at_round(bracket: &Bracket, round: usize, rng: &mut impl Rng) -> Option<RcTeam> {
        // Game indices by round:
        // Round 1: games 0-31 (32 winners)
        // Round 2: games 32-47 (16 winners)
        // Round 3: games 48-55 (8 winners - Sweet 16)
        // Round 4: games 56-59 (4 winners - Elite 8)
        // Round 5: games 60-61 (2 winners - Final Four)
        // Round 6: game 62 (1 winner - Champion)
        let (start, count) = match round {
            1 => (0, 32),
            2 => (32, 16),
            3 => (48, 8),
            4 => (56, 4),
            5 => (60, 2),
            6 => (62, 1),
            _ => return None,
        };

        if count == 0 {
            return None;
        }

        let game_idx = start + rng.gen_range(0..count);
        Some(Arc::clone(&bracket.games[game_idx].winner))
    }

    /// Run one generation of evolution
    pub fn evolve_generation(&mut self, tournament: &TournamentInfo) {
        let mut rng = rand::thread_rng();
        let mut new_population: Vec<Individual> = Vec::with_capacity(self.settings.population_size);

        // Elitism: keep top individuals
        let mut sorted_pop = self.population.clone();
        sorted_pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        for i in 0..self.settings.elitism_count.min(self.population.len()) {
            new_population.push(sorted_pop[i].clone());
        }

        // Generate rest of population
        while new_population.len() < self.settings.population_size {
            // Selection
            let parent1 = self.tournament_select(&mut rng);
            let parent2 = self.tournament_select(&mut rng);

            // Crossover
            let mut child = if rng.gen::<f64>() < self.settings.crossover_rate {
                Self::crossover(&parent1.bracket, &parent2.bracket, tournament, &self.scoring_config, &mut rng)
            } else {
                parent1.bracket.clone()
            };

            // Mutation: Always use Team-Round mutation (TeamRoundMutator)
            // Bit-flip mutation is semantically broken for brackets because
            // flipping an early round bit cascades unpredictably to later rounds
            if rng.gen::<f64>() < self.settings.mutation_rate {
                child = TeamRoundMutator::mutate(&child, tournament, &self.scoring_config);
            }

            let child = self.locks.repair(child, tournament, &self.scoring_config);
            new_population.push(Individual::new(child));
        }

        self.population = new_population;
        self.generation += 1;
    }

    /// Run the full GA optimization
    pub fn run(
        &mut self,
        tournament: &TournamentInfo,
        pool: &MonteCarloScenarios,
        verbose: bool,
    ) -> Bracket {
        for gen in 0..self.settings.generations {
            self.evaluate_fitness(pool);

            if verbose && gen % 20 == 0 {
                println!(
                    "Generation {}: Best fitness = {:.2}, Avg fitness = {:.2}",
                    gen,
                    self.best_fitness,
                    self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population.len() as f64
                );
            }

            self.evolve_generation(tournament);
        }

        // Final evaluation
        self.evaluate_fitness(pool);

        if verbose {
            println!(
                "Final: Best fitness = {:.2}",
                self.best_fitness
            );
        }

        self.best_bracket.clone().unwrap_or_else(|| {
            self.population[0].bracket.clone()
        })
    }

    /// Run GA for portfolio mode (optimizing best-ball contribution)
    /// For empty portfolio: fitness = average score (EV)
    /// For non-empty portfolio: fitness = marginal contribution to best-ball score
    pub fn run_for_portfolio(
        &mut self,
        tournament: &TournamentInfo,
        pool: &MonteCarloScenarios,
        existing_portfolio: &[Bracket],
        verbose: bool,
    ) -> Bracket {
        let fitness_label = if existing_portfolio.is_empty() {
            "Best-ball score"
        } else {
            "Marginal contribution"
        };

        for gen in 0..self.settings.generations {
            self.evaluate_fitness_portfolio(pool, existing_portfolio);

            if verbose && gen % 20 == 0 {
                println!(
                    "Generation {}: {} = {:.2}",
                    gen,
                    fitness_label,
                    self.best_fitness
                );
            }

            self.evolve_generation(tournament);
        }

        self.evaluate_fitness_portfolio(pool, existing_portfolio);

        if verbose {
            println!(
                "Final: {} = {:.2}",
                fitness_label,
                self.best_fitness
            );
        }

        self.best_bracket.clone().unwrap_or_else(|| {
            self.population[0].bracket.clone()
        })
    }
}

/// Sequential Portfolio Optimizer
/// Optimizes brackets one at a time, freezing each before moving to the next
/// Uses SA-style approach: optimize bracket 1, freeze, optimize bracket 2 for marginal contribution, etc.
pub struct SequentialPortfolioOptimizer {
    pub config: Config,
    pub scoring_config: ScoringConfig,
    pub locks: LockSet,
}

/// Portfolio Individual - represents an entire portfolio of N brackets
#[derive(Clone)]
pub struct PortfolioIndividual {
    pub brackets: Vec<Bracket>,
    pub fitness: f64,
}

impl PortfolioIndividual {
    pub fn new(brackets: Vec<Bracket>) -> Self {
        PortfolioIndividual {
            brackets,
            fitness: 0.0,
        }
    }

    pub fn random(tournament: &TournamentInfo, num_brackets: usize, scoring_config: &ScoringConfig) -> Self {
        let brackets: Vec<Bracket> = (0..num_brackets)
            .map(|_| Bracket::new(tournament, Some(scoring_config)))
            .collect();
        PortfolioIndividual::new(brackets)
    }
}

/// Whole Portfolio GA - evolves entire portfolios at once
/// Each individual in the population is a complete portfolio of N brackets
/// Fitness is best-ball score across all simulations
pub struct WholePortfolioGA {
    pub population: Vec<PortfolioIndividual>,
    pub settings: GaSettings,
    pub scoring_config: ScoringConfig,
    pub num_brackets: usize,
    pub generation: usize,
    pub best_fitness: f64,
    pub best_portfolio: Option<Vec<Bracket>>,
    /// Applied to every bracket of every portfolio.
    pub locks: LockSet,
}

impl WholePortfolioGA {
    pub fn new(
        tournament: &TournamentInfo,
        num_brackets: usize,
        settings: GaSettings,
        scoring_config: ScoringConfig,
    ) -> Self {
        // Initialize population of portfolios
        let population: Vec<PortfolioIndividual> = (0..settings.population_size)
            .into_par_iter()
            .map(|_| PortfolioIndividual::random(tournament, num_brackets, &scoring_config))
            .collect();

        WholePortfolioGA {
            population,
            settings,
            scoring_config,
            num_brackets,
            generation: 0,
            best_fitness: 0.0,
            best_portfolio: None,
            locks: LockSet::default(),
        }
    }

    /// Constrain every bracket in every portfolio to satisfy these locks.
    pub fn with_locks(mut self, locks: LockSet, tournament: &TournamentInfo) -> Self {
        if !locks.is_empty() {
            let scoring = self.scoring_config;
            self.population = self
                .population
                .drain(..)
                .map(|ind| {
                    PortfolioIndividual::new(
                        ind.brackets
                            .into_iter()
                            .map(|b| locks.repair(b, tournament, &scoring))
                            .collect(),
                    )
                })
                .collect();
        }
        self.locks = locks;
        self
    }

    /// Evaluate fitness for all portfolios using best-ball scoring
    pub fn evaluate_fitness(&mut self, pool: &MonteCarloScenarios) {
        let fitnesses: Vec<f64> = self.population
            .par_iter()
            .map(|ind| pool.score_portfolio_best_ball(&ind.brackets, &self.scoring_config))
            .collect();

        for (ind, fitness) in self.population.iter_mut().zip(fitnesses.into_iter()) {
            ind.fitness = fitness;
        }

        // Track best
        if let Some(best) = self.population.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap()
        }) {
            if best.fitness > self.best_fitness {
                self.best_fitness = best.fitness;
                self.best_portfolio = Some(best.brackets.clone());
            }
        }
    }

    /// Tournament selection for portfolios
    fn tournament_select(&self, rng: &mut impl Rng) -> &PortfolioIndividual {
        let mut best: Option<&PortfolioIndividual> = None;

        for _ in 0..self.settings.tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            let candidate = &self.population[idx];

            if best.is_none() || candidate.fitness > best.unwrap().fitness {
                best = Some(candidate);
            }
        }

        best.unwrap()
    }

    /// Crossover two portfolios using Team-Round crossover on each bracket
    /// For each bracket position, do Team-Round crossover of the two parent brackets
    fn crossover(
        parent1: &PortfolioIndividual,
        parent2: &PortfolioIndividual,
        tournament: &TournamentInfo,
        scoring_config: &ScoringConfig,
        rng: &mut impl Rng,
    ) -> PortfolioIndividual {
        let mut child_brackets = Vec::with_capacity(parent1.brackets.len());

        for i in 0..parent1.brackets.len() {
            // Team-Round crossover for each bracket position
            let child_bracket = Self::team_round_crossover_bracket(
                &parent1.brackets[i],
                &parent2.brackets[i],
                tournament,
                scoring_config,
                rng,
            );
            child_brackets.push(child_bracket);
        }

        PortfolioIndividual::new(child_brackets)
    }

    /// Team-Round crossover for a single bracket
    /// Takes Team-Round pairs from donor and applies them to base
    fn team_round_crossover_bracket(
        parent1: &Bracket,
        parent2: &Bracket,
        tournament: &TournamentInfo,
        scoring_config: &ScoringConfig,
        rng: &mut impl Rng,
    ) -> Bracket {
        // Pick which parent is the base (50/50)
        let (base, donor) = if rng.gen::<bool>() {
            (parent1, parent2)
        } else {
            (parent2, parent1)
        };

        let mut child_binary = base.binary.clone();

        // Pick N Team-Round pairs from donor to inject (N = 1 to 4)
        let num_injections = rng.gen_range(1..=4);

        for _ in 0..num_injections {
            let round: usize = rng.gen_range(1..=6);

            if let Some(team) = Self::get_team_at_round(donor, round, rng) {
                child_binary = TeamRoundMutator::force_team_to_round(
                    &child_binary,
                    tournament,
                    &team,
                    round,
                );
            }
        }

        Bracket::new_from_binary(tournament, &child_binary, Some(scoring_config))
    }

    /// Get a random team that reached a specific round in the bracket
    fn get_team_at_round(bracket: &Bracket, round: usize, rng: &mut impl Rng) -> Option<RcTeam> {
        let (start, count) = match round {
            1 => (0, 32),
            2 => (32, 16),
            3 => (48, 8),
            4 => (56, 4),
            5 => (60, 2),
            6 => (62, 1),
            _ => return None,
        };

        if count == 0 {
            return None;
        }

        let game_idx = start + rng.gen_range(0..count);
        Some(Arc::clone(&bracket.games[game_idx].winner))
    }

    /// Mutate a portfolio - apply smart mutation to one random bracket
    fn mutate(
        portfolio: &PortfolioIndividual,
        tournament: &TournamentInfo,
        scoring_config: &ScoringConfig,
        rng: &mut impl Rng,
    ) -> PortfolioIndividual {
        let mut new_brackets = portfolio.brackets.clone();

        // Pick a random bracket to mutate
        let idx = rng.gen_range(0..new_brackets.len());

        // Apply smart mutation
        new_brackets[idx] = TeamRoundMutator::mutate(&new_brackets[idx], tournament, scoring_config);

        PortfolioIndividual::new(new_brackets)
    }

    /// Run one generation of evolution
    pub fn evolve_generation(&mut self, tournament: &TournamentInfo) {
        let mut rng = rand::thread_rng();
        let mut new_population: Vec<PortfolioIndividual> = Vec::with_capacity(self.settings.population_size);

        // Elitism: keep top portfolios
        let mut sorted_pop = self.population.clone();
        sorted_pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        for i in 0..self.settings.elitism_count.min(self.population.len()) {
            new_population.push(sorted_pop[i].clone());
        }

        // Generate rest of population
        while new_population.len() < self.settings.population_size {
            let parent1 = self.tournament_select(&mut rng);
            let parent2 = self.tournament_select(&mut rng);

            // Crossover using region-based bracket crossover
            let mut child = if rng.gen::<f64>() < self.settings.crossover_rate {
                Self::crossover(parent1, parent2, tournament, &self.scoring_config, &mut rng)
            } else {
                parent1.clone()
            };

            // Mutation: always use Team-Round mutation (TeamRoundMutator)
            if rng.gen::<f64>() < self.settings.mutation_rate {
                child = Self::mutate(&child, tournament, &self.scoring_config, &mut rng);
            }

            let child = if self.locks.is_empty() {
                child
            } else {
                PortfolioIndividual::new(
                    child
                        .brackets
                        .into_iter()
                        .map(|b| self.locks.repair(b, tournament, &self.scoring_config))
                        .collect(),
                )
            };
            new_population.push(child);
        }

        self.population = new_population;
        self.generation += 1;
    }

    /// Run the full GA optimization
    pub fn run(
        &mut self,
        tournament: &TournamentInfo,
        pool: &MonteCarloScenarios,
        verbose: bool,
    ) -> Vec<Bracket> {
        for gen in 0..self.settings.generations {
            self.evaluate_fitness(pool);

            if verbose && gen % 20 == 0 {
                let avg_fitness = self.population.iter().map(|i| i.fitness).sum::<f64>()
                    / self.population.len() as f64;
                println!(
                    "Generation {}: Best = {:.2}, Avg = {:.2}",
                    gen, self.best_fitness, avg_fitness
                );
            }

            self.evolve_generation(tournament);
        }

        // Final evaluation
        self.evaluate_fitness(pool);

        if verbose {
            println!("Final: Best fitness = {:.2}", self.best_fitness);
        }

        self.best_portfolio.clone().unwrap_or_else(|| {
            self.population[0].brackets.clone()
        })
    }
}

impl SequentialPortfolioOptimizer {
    /// `scoring_config` is passed in rather than re-derived from `config`.
    /// Deriving it here meant this optimizer silently read the YAML scoring
    /// section while the rest of the program used the `--score-r*` flags, so
    /// the same flags changed the answer in some modes and not others.
    pub fn new(config: Config, scoring_config: ScoringConfig, locks: LockSet) -> Self {
        SequentialPortfolioOptimizer {
            config,
            scoring_config,
            locks,
        }
    }

    /// Optimize a portfolio of brackets sequentially
    pub fn optimize(
        &self,
        tournament: &TournamentInfo,
        num_brackets: usize,
        verbose: bool,
    ) -> Vec<Bracket> {
        // Generate simulation pool once
        let pool = MonteCarloScenarios::new(
            tournament,
            self.config.simulation.pool_size,
            &self.scoring_config,
        );

        let mut portfolio: Vec<Bracket> = Vec::with_capacity(num_brackets);

        for i in 0..num_brackets {
            println!("\n=== Optimizing Bracket {} of {} ===", i + 1, num_brackets);

            // Create new GA instance
            let mut ga = GeneticAlgorithm::new(
                tournament,
                self.config.ga.clone(),
                self.scoring_config,
            )
            .with_locks(self.locks.clone(), tournament);

            // Always optimize for best-ball contribution to portfolio
            // For first bracket, this is equivalent to EV, but keeps the fitness semantics consistent
            // For subsequent brackets, this is marginal contribution to existing portfolio
            let bracket = ga.run_for_portfolio(tournament, &pool, &portfolio, verbose);

            // Calculate and display best-ball score
            let portfolio_with_new: Vec<Bracket> = portfolio.iter()
                .chain(std::iter::once(&bracket))
                .cloned()
                .collect();

            let best_ball_score = pool.score_portfolio_best_ball(&portfolio_with_new, &self.scoring_config);

            println!(
                "Bracket {}: Champion = {} (seed {}), EV = {:.2}",
                i + 1,
                bracket.winner.name,
                bracket.winner.seed,
                bracket.expected_value
            );
            println!("Portfolio best-ball score after bracket {}: {:.2}", i + 1, best_ball_score);

            portfolio.push(bracket);
        }

        // Final summary
        println!("\n=== Portfolio Optimization Complete ===");
        let final_score = pool.score_portfolio_best_ball(&portfolio, &self.scoring_config);
        println!("Final portfolio best-ball score: {:.2}", final_score);

        // Show individual bracket scores for comparison
        println!("\nIndividual bracket scores:");
        for (i, bracket) in portfolio.iter().enumerate() {
            let individual_score = pool.score_bracket(bracket, &self.scoring_config);
            println!(
                "  Bracket {}: {} - Score: {:.2}, EV: {:.2}",
                i + 1,
                bracket.winner.name,
                individual_score,
                bracket.expected_value
            );
        }

        portfolio
    }
}

/// Hybrid Simulated Annealing + GA for single bracket optimization
pub struct HybridOptimizer {
    pub config: Config,
    pub scoring_config: ScoringConfig,
    pub locks: LockSet,
}

impl HybridOptimizer {
    /// See `SequentialPortfolioOptimizer::new` on why scoring is passed in.
    pub fn new(config: Config, scoring_config: ScoringConfig, locks: LockSet) -> Self {
        HybridOptimizer {
            config,
            scoring_config,
            locks,
        }
    }

    /// Run hybrid SA+GA optimization on a single bracket
    /// Uses SA acceptance criterion with GA-style operators
    pub fn optimize_single(
        &self,
        tournament: &TournamentInfo,
        verbose: bool,
    ) -> Bracket {
        let pool = MonteCarloScenarios::new(
            tournament,
            self.config.simulation.pool_size,
            &self.scoring_config,
        );

        let mut rng = rand::thread_rng();

        // Start with a random bracket
        let mut current = self.locks.repair(
            Bracket::new(tournament, Some(&self.scoring_config)),
            tournament,
            &self.scoring_config,
        );
        let mut current_score = pool.score_bracket(&current, &self.scoring_config);

        let mut best = current.clone();
        let mut best_score = current_score;

        // SA parameters
        let initial_temp: f64 = 10.0;
        let final_temp: f64 = 0.1;
        let cooling_rate = (final_temp / initial_temp).powf(1.0 / self.config.ga.generations as f64);
        let mut temperature = initial_temp;

        for gen in 0..self.config.ga.generations {
            // Generate neighbor using Team-Round mutation (TeamRoundMutator)
            // Bit-flip mutation is semantically broken for brackets
            let neighbor = self.locks.repair(
                TeamRoundMutator::mutate(&current, tournament, &self.scoring_config),
                tournament,
                &self.scoring_config,
            );

            let neighbor_score = pool.score_bracket(&neighbor, &self.scoring_config);

            // SA acceptance criterion
            let accept = if neighbor_score > current_score {
                true
            } else {
                let delta = neighbor_score - current_score;
                let accept_prob = (delta / temperature).exp();
                rng.gen::<f64>() < accept_prob
            };

            if accept {
                current = neighbor;
                current_score = neighbor_score;

                if current_score > best_score {
                    best = current.clone();
                    best_score = current_score;
                }
            }

            temperature *= cooling_rate;

            if verbose && gen % 50 == 0 {
                println!(
                    "Generation {}: Current = {:.2}, Best = {:.2}, Temp = {:.4}",
                    gen, current_score, best_score, temperature
                );
            }
        }

        if verbose {
            println!("Final best score: {:.2}", best_score);
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bracket::tests::assert_legal;
    use crate::ingest::tests::tournament;

    #[test]
    fn forcing_a_team_to_a_round_actually_gets_it_there() {
        // The old implementation set each bit from the moving team's own seed
        // (`team.seed <= 8`), ignoring the opponent, so about half of these
        // forced the team to lose instead. Check every team at every depth.
        let t = tournament();
        let start = Bracket::new(&t, None);

        for team_index in 0..64u8 {
            for wins in 1..=6usize {
                let binary =
                    TeamRoundMutator::force_index_to_round(&start.binary, &t, team_index, wins);
                let bracket = Bracket::new_from_binary(&t, &binary, None);

                assert_legal(&bracket);
                assert_eq!(
                    bracket.wins_for(team_index),
                    wins.max(bracket.wins_for(team_index)),
                    "team {} was asked for {} wins",
                    team_index,
                    wins
                );
                assert!(
                    bracket.wins_for(team_index) >= wins,
                    "team {} only won {} of the {} games it was forced to win",
                    team_index,
                    bracket.wins_for(team_index),
                    wins
                );
                if wins == 6 {
                    assert_eq!(bracket.winner.team_index, team_index);
                }
            }
        }
    }

    #[test]
    fn forcing_a_champion_works_from_any_starting_bracket() {
        let t = tournament();
        for _ in 0..25 {
            let start = Bracket::new(&t, None);
            for team_index in [0u8, 15, 16, 31, 32, 47, 48, 63] {
                let binary =
                    TeamRoundMutator::force_index_to_round(&start.binary, &t, team_index, 6);
                let bracket = Bracket::new_from_binary(&t, &binary, None);
                assert_legal(&bracket);
                assert_eq!(bracket.winner.team_index, team_index);
            }
        }
    }

    #[test]
    fn forcing_is_idempotent() {
        let t = tournament();
        let start = Bracket::new(&t, None);
        let once = TeamRoundMutator::force_index_to_round(&start.binary, &t, 20, 4);
        let twice = TeamRoundMutator::force_index_to_round(&once, &t, 20, 4);
        assert_eq!(once, twice);
    }

    #[test]
    fn forcing_only_touches_the_teams_own_path() {
        // Games in the opposite half of the draw must be left alone.
        let t = tournament();
        let start = Bracket::new(&t, None);
        let team_index = t.r1_teams[0][0];
        let binary = TeamRoundMutator::force_index_to_round(&start.binary, &t, team_index, 4);

        let mut path = vec![t.r1_game_of_team[team_index as usize]];
        while let Some(&g) = path.last() {
            if crate::tree::PARENT[g] == NO_GAME || path.len() >= 4 {
                break;
            }
            path.push(crate::tree::PARENT[g]);
        }

        for game in 0..crate::tree::NUM_GAMES {
            if !path.contains(&game) {
                assert_eq!(
                    binary[game], start.binary[game],
                    "game {} changed but is not on the forced path {:?}",
                    game, path
                );
            }
        }
    }

    #[test]
    fn mutation_always_produces_a_legal_bracket() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let mut bracket = Bracket::new(&t, Some(&scoring));
        for _ in 0..300 {
            bracket = TeamRoundMutator::mutate(&bracket, &t, &scoring);
            assert_legal(&bracket);
        }
    }

    #[test]
    fn locks_survive_repeated_mutation() {
        // A lock applied only to the starting bracket used to be discarded by
        // the first mutation, because mutation rewrites the bit vector.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let champion = t.teams.iter().find(|x| x.seed == 11).unwrap().team_index;
        let dark_horse = t
            .teams
            .iter()
            .find(|x| x.seed == 13 && t.region_rank[x.team_index as usize] != t.region_rank[champion as usize])
            .unwrap()
            .team_index;

        let locks = LockSet::new(vec![
            TeamLock { team_index: champion, wins_required: 6 },
            TeamLock { team_index: dark_horse, wins_required: 3 },
        ]);

        let mut bracket = locks.repair(Bracket::new(&t, Some(&scoring)), &t, &scoring);
        for _ in 0..200 {
            bracket = locks.repair(
                TeamRoundMutator::mutate(&bracket, &t, &scoring),
                &t,
                &scoring,
            );
            assert_legal(&bracket);
            assert!(locks.is_satisfied(&bracket));
            assert_eq!(bracket.winner.team_index, champion);
        }
    }

    #[test]
    fn crossover_children_are_legal() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let a = Bracket::new(&t, Some(&scoring));
            let b = Bracket::new(&t, Some(&scoring));
            assert_legal(&GeneticAlgorithm::crossover(&a, &b, &t, &scoring, &mut rng));
        }
    }
}
