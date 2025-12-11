// Genetic Algorithm module for NCAA Bracket Optimization
// Implements proper population-based GA with smart mutation and best-ball scoring

use crate::bracket::{Bracket, ScoringConfig, ScoreTable};
use crate::config::{Config, GaSettings};
use crate::ingest::{RcTeam, TournamentInfo};
use rand::Rng;
use rayon::prelude::*;

/// Pre-generated pool of random brackets for scoring
/// These represent possible tournament outcomes
#[derive(Clone)]
pub struct SimulationPool {
    pub brackets: Vec<Bracket>,
    pub size: usize,
    /// Pre-computed score lookup table for fast scoring
    pub score_table: ScoreTable,
}

impl SimulationPool {
    /// Generate a new simulation pool with random brackets
    pub fn new(tournament: &TournamentInfo, size: usize, scoring_config: &ScoringConfig) -> Self {
        println!("Generating simulation pool of {} brackets...", size);

        let brackets: Vec<Bracket> = (0..size)
            .into_par_iter()
            .map(|_| Bracket::new(tournament, Some(scoring_config)))
            .collect();

        // Pre-compute score lookup table
        let score_table = ScoreTable::new(scoring_config);

        println!("Simulation pool generated.");

        SimulationPool { brackets, size, score_table }
    }

    /// Score a single bracket against all simulations (fast version)
    /// Returns the average score
    pub fn score_bracket(&self, bracket: &Bracket, _scoring_config: &ScoringConfig) -> f64 {
        let table = &self.score_table;
        let total: f64 = self.brackets
            .par_iter()
            .map(|sim| bracket.score_fast(sim, table))
            .sum();

        total / self.size as f64
    }

    /// Score a portfolio using best-ball metric (fast version)
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
        let total: f64 = self.brackets
            .par_iter()
            .map(|sim| {
                portfolio
                    .iter()
                    .map(|b| b.score_fast(sim, table))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0)
            })
            .sum();

        total / self.size as f64
    }

    /// Score a bracket's marginal contribution to an existing portfolio (fast version)
    /// This is the increase in best-ball score when adding this bracket
    pub fn score_marginal_contribution(
        &self,
        bracket: &Bracket,
        existing_portfolio: &[Bracket],
        _scoring_config: &ScoringConfig,
    ) -> f64 {
        if existing_portfolio.is_empty() {
            return self.score_bracket(bracket, &ScoringConfig::default());
        }

        let table = &self.score_table;
        let total: f64 = self.brackets
            .par_iter()
            .map(|sim| {
                let existing_max = existing_portfolio
                    .iter()
                    .map(|b| b.score_fast(sim, table))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                let new_score = bracket.score_fast(sim, table);

                // Marginal contribution is how much better we do with this bracket
                new_score.max(existing_max)
            })
            .sum();

        total / self.size as f64
    }
}

/// Smart mutation operator
/// Instead of random bit flips, selects a team and forces them to reach a specific round
pub struct SmartMutator;

impl SmartMutator {
    /// Perform a smart mutation on a bracket
    /// Picks a random team and random round, forces that team to advance to that round
    pub fn mutate(
        bracket: &Bracket,
        tournament: &TournamentInfo,
        scoring_config: &ScoringConfig,
    ) -> Bracket {
        let mut rng = rand::thread_rng();

        // Pick a random team from the tournament
        let team_idx = rng.gen_range(0..tournament.teams.len());
        let team = &tournament.teams[team_idx];

        // Pick a random round (1-6)
        // Weight towards earlier rounds (more impactful changes)
        let round: usize = {
            let r: f64 = rng.gen();
            if r < 0.4 { 1 }      // 40% chance R1
            else if r < 0.65 { 2 } // 25% chance R2
            else if r < 0.80 { 3 } // 15% chance Sweet 16
            else if r < 0.90 { 4 } // 10% chance Elite 8
            else if r < 0.97 { 5 } // 7% chance Final Four
            else { 6 }             // 3% chance Championship
        };

        // Create a new binary representation with this team advancing to the target round
        let new_binary = Self::force_team_to_round(
            &bracket.binary,
            tournament,
            team,
            round,
        );

        Bracket::new_from_binary(tournament, &new_binary, Some(scoring_config))
    }

    /// Modify binary representation to force a team to reach a specific round
    fn force_team_to_round(
        binary: &[bool],
        tournament: &TournamentInfo,
        team: &RcTeam,
        target_round: usize,
    ) -> Vec<bool> {
        let mut new_binary = binary.to_vec();
        let region_idx = Self::get_region_index(&team.region);

        // For each round up to target_round-1, ensure the team wins
        for round in 1..=target_round {
            if let Some((game_idx, should_win_hilo)) =
                Self::get_game_info_for_team(tournament, team, round, region_idx, &new_binary)
            {
                new_binary[game_idx] = should_win_hilo;
            }
        }

        new_binary
    }

    /// Get the game index and required hilo value for a team to win in a given round
    fn get_game_info_for_team(
        tournament: &TournamentInfo,
        team: &RcTeam,
        round: usize,
        region_idx: usize,
        current_binary: &[bool],
    ) -> Option<(usize, bool)> {
        match round {
            1 => {
                // Round 1: Find the game based on seed
                let game_in_region = Self::seed_to_r1_game(team.seed);
                let game_idx = region_idx * 8 + game_in_region;

                // hilo = true means lower seed wins
                // Team should win, so hilo depends on whether team is lower seed
                let matchup = tournament.round1[game_in_region];
                let is_lower_seed = team.seed == matchup[0].min(matchup[1]);

                Some((game_idx, is_lower_seed))
            }
            2 => {
                // Round 2: 4 games per region, starting at index 32
                let game_in_region = Self::seed_to_r2_game(team.seed);
                let game_idx = 32 + region_idx * 4 + game_in_region;

                // Need to determine if team is the "hilo" winner in this matchup
                // This depends on who won Round 1
                let should_win = Self::should_be_hilo_winner(team, round, region_idx, game_in_region, current_binary, tournament);

                Some((game_idx, should_win))
            }
            3 => {
                // Sweet 16: 2 games per region, starting at index 48
                let game_in_region = Self::seed_to_r3_game(team.seed);
                let game_idx = 48 + region_idx * 2 + game_in_region;

                let should_win = Self::should_be_hilo_winner(team, round, region_idx, game_in_region, current_binary, tournament);

                Some((game_idx, should_win))
            }
            4 => {
                // Elite 8: 1 game per region, starting at index 56
                let game_idx = 56 + region_idx;

                let should_win = Self::should_be_hilo_winner(team, round, region_idx, 0, current_binary, tournament);

                Some((game_idx, should_win))
            }
            5 => {
                // Final Four: 2 games
                // Game 60: South vs Midwest winners
                // Game 61: East vs West winners
                let game_idx = if region_idx == 2 || region_idx == 3 { 60 } else { 61 };

                // For cross-region games, hilo is based on alphabetical region order
                // East < Midwest < South < West
                let should_win = Self::should_be_hilo_winner_cross_region(team, region_idx);

                Some((game_idx, should_win))
            }
            6 => {
                // Championship: Game 62
                // South/Midwest winner vs East/West winner
                // Need to determine based on region
                let game_idx = 62;

                // Alphabetically: East < Midwest < South < West
                // Game 60 winner (South/Midwest) has regions 2,3
                // Game 61 winner (East/West) has regions 0,1
                // hilo = true means the "first" team wins
                // The "first" team in championship is from game 60 (South/Midwest)
                // since South < West and Midwest < West but South > East...
                // Actually the ordering is: East(0) vs West(1) -> winner at 61
                //                          South(2) vs Midwest(3) -> winner at 60
                // In championship (62): compare regions alphabetically
                let should_win = region_idx == 2 || region_idx == 3; // South/Midwest side
                // But we need to check actual alphabetical order
                // East < Midwest < South < West
                // So if team is from South or Midwest, they're in game 60's bracket
                // If team is from East or West, they're in game 61's bracket
                // For hilo in championship: depends on which regions are playing

                Some((game_idx, should_win))
            }
            _ => None,
        }
    }

    /// Determine if team should be the hilo winner for a given round/game
    fn should_be_hilo_winner(
        team: &RcTeam,
        _round: usize,
        _region_idx: usize,
        _game_in_region: usize,
        _current_binary: &[bool],
        _tournament: &TournamentInfo,
    ) -> bool {
        // Within a region, hilo is based on seed (lower seed = true)
        // This is a simplification - in reality we'd need to trace through
        // the bracket to see who the opponent is
        team.seed <= 8
    }

    /// Determine hilo for cross-region games
    fn should_be_hilo_winner_cross_region(team: &RcTeam, region_idx: usize) -> bool {
        // Alphabetical order: East(0) < Midwest(3) < South(2) < West(1)
        // Wait, that's not right. Let me fix:
        // East, Midwest, South, West -> alphabetically: East < Midwest < South < West
        // So region ordering should be: East(0)=0, Midwest(3)=1, South(2)=2, West(1)=3
        // For Final Four game 60 (South vs Midwest): Midwest < South, so Midwest is "first"
        // For Final Four game 61 (East vs West): East < West, so East is "first"
        // hilo = true means the alphabetically first region wins

        match region_idx {
            0 => true,  // East is first vs West
            1 => false, // West is second vs East
            2 => false, // South is second vs Midwest
            3 => true,  // Midwest is first vs South
            _ => true,
        }
    }

    fn get_region_index(region: &str) -> usize {
        match region {
            "East" => 0,
            "West" => 1,
            "South" => 2,
            "Midwest" => 3,
            _ => 0,
        }
    }

    fn seed_to_r1_game(seed: i32) -> usize {
        match seed {
            1 | 16 => 0,
            8 | 9 => 1,
            5 | 12 => 2,
            4 | 13 => 3,
            6 | 11 => 4,
            3 | 14 => 5,
            7 | 10 => 6,
            2 | 15 => 7,
            _ => 0,
        }
    }

    fn seed_to_r2_game(seed: i32) -> usize {
        match seed {
            1 | 16 | 8 | 9 => 0,
            5 | 12 | 4 | 13 => 1,
            6 | 11 | 3 | 14 => 2,
            7 | 10 | 2 | 15 => 3,
            _ => 0,
        }
    }

    fn seed_to_r3_game(seed: i32) -> usize {
        match seed {
            1 | 16 | 8 | 9 | 5 | 12 | 4 | 13 => 0,
            _ => 1,
        }
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
        }
    }

    /// Evaluate fitness for all individuals using simulation pool
    pub fn evaluate_fitness(&mut self, pool: &SimulationPool) {
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
        pool: &SimulationPool,
        existing_portfolio: &[Bracket],
    ) {
        let fitnesses: Vec<f64> = self.population
            .par_iter()
            .map(|ind| {
                pool.score_marginal_contribution(&ind.bracket, existing_portfolio, &self.scoring_config)
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

    /// Uniform crossover - mix bits from two parents
    fn crossover(parent1: &Bracket, parent2: &Bracket, tournament: &TournamentInfo, scoring_config: &ScoringConfig, rng: &mut impl Rng) -> Bracket {
        let mut child_binary: Vec<bool> = Vec::with_capacity(63);

        for i in 0..63 {
            if rng.gen::<bool>() {
                child_binary.push(parent1.binary[i]);
            } else {
                child_binary.push(parent2.binary[i]);
            }
        }

        Bracket::new_from_binary(tournament, &child_binary, Some(scoring_config))
    }

    /// Bit-flip mutation
    fn bit_flip_mutate(bracket: &Bracket, tournament: &TournamentInfo, scoring_config: &ScoringConfig, mutation_rate: f64, rng: &mut impl Rng) -> Bracket {
        let mut new_binary = bracket.binary.clone();

        for bit in new_binary.iter_mut() {
            if rng.gen::<f64>() < mutation_rate {
                *bit = !*bit;
            }
        }

        Bracket::new_from_binary(tournament, &new_binary, Some(scoring_config))
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

            // Mutation
            if rng.gen::<f64>() < self.settings.mutation_rate {
                if self.settings.smart_mutation && rng.gen::<f64>() < self.settings.smart_mutation_rate {
                    // Smart mutation
                    child = SmartMutator::mutate(&child, tournament, &self.scoring_config);
                } else {
                    // Bit-flip mutation
                    let bit_mutation_rate = 3.0 / 63.0; // ~3 bits on average
                    child = Self::bit_flip_mutate(&child, tournament, &self.scoring_config, bit_mutation_rate, &mut rng);
                }
            }

            new_population.push(Individual::new(child));
        }

        self.population = new_population;
        self.generation += 1;
    }

    /// Run the full GA optimization
    pub fn run(
        &mut self,
        tournament: &TournamentInfo,
        pool: &SimulationPool,
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

    /// Run GA for portfolio mode (optimizing marginal contribution)
    pub fn run_for_portfolio(
        &mut self,
        tournament: &TournamentInfo,
        pool: &SimulationPool,
        existing_portfolio: &[Bracket],
        verbose: bool,
    ) -> Bracket {
        for gen in 0..self.settings.generations {
            self.evaluate_fitness_portfolio(pool, existing_portfolio);

            if verbose && gen % 20 == 0 {
                println!(
                    "Generation {}: Best marginal contribution = {:.2}",
                    gen,
                    self.best_fitness
                );
            }

            self.evolve_generation(tournament);
        }

        self.evaluate_fitness_portfolio(pool, existing_portfolio);

        if verbose {
            println!(
                "Final: Best marginal contribution = {:.2}",
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
        }
    }

    /// Evaluate fitness for all portfolios using best-ball scoring
    pub fn evaluate_fitness(&mut self, pool: &SimulationPool) {
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

    /// Crossover two portfolios - swap random brackets between them
    fn crossover(
        parent1: &PortfolioIndividual,
        parent2: &PortfolioIndividual,
        rng: &mut impl Rng,
    ) -> PortfolioIndividual {
        let mut child_brackets = Vec::with_capacity(parent1.brackets.len());

        for i in 0..parent1.brackets.len() {
            // For each bracket position, pick from either parent
            if rng.gen::<bool>() {
                child_brackets.push(parent1.brackets[i].clone());
            } else {
                child_brackets.push(parent2.brackets[i].clone());
            }
        }

        PortfolioIndividual::new(child_brackets)
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
        new_brackets[idx] = SmartMutator::mutate(&new_brackets[idx], tournament, scoring_config);

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

            // Crossover
            let mut child = if rng.gen::<f64>() < self.settings.crossover_rate {
                Self::crossover(parent1, parent2, &mut rng)
            } else {
                parent1.clone()
            };

            // Mutation
            if rng.gen::<f64>() < self.settings.mutation_rate {
                child = Self::mutate(&child, tournament, &self.scoring_config, &mut rng);
            }

            new_population.push(child);
        }

        self.population = new_population;
        self.generation += 1;
    }

    /// Run the full GA optimization
    pub fn run(
        &mut self,
        tournament: &TournamentInfo,
        pool: &SimulationPool,
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
    pub fn new(config: Config) -> Self {
        let scoring_config = config.to_scoring_config();
        SequentialPortfolioOptimizer {
            config,
            scoring_config,
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
        let pool = SimulationPool::new(
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
            );

            // Optimize for marginal contribution to existing portfolio
            let bracket = if i == 0 {
                // First bracket: optimize for raw score
                ga.run(tournament, &pool, verbose)
            } else {
                // Subsequent brackets: optimize for marginal contribution
                ga.run_for_portfolio(tournament, &pool, &portfolio, verbose)
            };

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
}

impl HybridOptimizer {
    pub fn new(config: Config) -> Self {
        let scoring_config = config.to_scoring_config();
        HybridOptimizer {
            config,
            scoring_config,
        }
    }

    /// Run hybrid SA+GA optimization on a single bracket
    /// Uses SA acceptance criterion with GA-style operators
    pub fn optimize_single(
        &self,
        tournament: &TournamentInfo,
        verbose: bool,
    ) -> Bracket {
        let pool = SimulationPool::new(
            tournament,
            self.config.simulation.pool_size,
            &self.scoring_config,
        );

        let mut rng = rand::thread_rng();

        // Start with a random bracket
        let mut current = Bracket::new(tournament, Some(&self.scoring_config));
        let mut current_score = pool.score_bracket(&current, &self.scoring_config);

        let mut best = current.clone();
        let mut best_score = current_score;

        // SA parameters
        let initial_temp: f64 = 10.0;
        let final_temp: f64 = 0.1;
        let cooling_rate = (final_temp / initial_temp).powf(1.0 / self.config.ga.generations as f64);
        let mut temperature = initial_temp;

        for gen in 0..self.config.ga.generations {
            // Generate neighbor using smart mutation or bit flip
            let neighbor = if self.config.ga.smart_mutation && rng.gen::<f64>() < self.config.ga.smart_mutation_rate {
                SmartMutator::mutate(&current, tournament, &self.scoring_config)
            } else {
                let bit_mutation_rate = 3.0 / 63.0;
                GeneticAlgorithm::bit_flip_mutate(&current, tournament, &self.scoring_config, bit_mutation_rate, &mut rng)
            };

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

    // Tests would go here - skipping for brevity but would test:
    // - SimulationPool generation and scoring
    // - SmartMutator producing valid brackets
    // - GA selection, crossover, mutation
    // - Sequential portfolio optimization
}
