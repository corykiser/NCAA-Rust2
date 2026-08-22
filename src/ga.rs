// Genetic Algorithm module for NCAA Bracket Optimization
// Implements proper population-based GA with smart mutation and best-ball scoring

use crate::bracket::{Bracket, ScoreTable, ScoringConfig};
use crate::config::{Config, GaSettings};
use crate::exact::TeamLock;
use crate::ingest::{RcTeam, TournamentInfo};
use crate::picks::Picks;
use crate::score::{ScenarioPool, ScoredPicks};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Fixed sample of tournament outcomes used to score candidates.
///
/// A thin, `Bracket`-shaped facade over [`ScenarioPool`], which holds the
/// scenarios as raw winner bytes. The pool used to keep 10,000 full `Bracket`
/// objects *and* a parallel array of `FastBracket` copies — around 75 MB and
/// two million atomic refcount operations to build something the scorer only
/// ever reads 63 bytes of.
///
/// New code should reach for [`MonteCarloScenarios::pool`] and work in
/// [`Picks`]; the `&Bracket` methods here exist for the callers that still hold
/// full brackets.
pub struct MonteCarloScenarios {
    pub pool: ScenarioPool,
    pub size: usize,
    /// Pre-computed score lookup table for fast scoring.
    pub score_table: ScoreTable,
}

/// Default scenario-pool seed. Fixed so that two runs of the program with the
/// same settings score candidates against the same tournaments and can be
/// compared directly; pass an explicit seed to vary it.
pub const DEFAULT_POOL_SEED: u64 = 0x4E_43_41_41_32_30_32_35;

impl MonteCarloScenarios {
    /// Generate a new simulation pool with random brackets.
    pub fn new(tournament: &TournamentInfo, size: usize, scoring_config: &ScoringConfig) -> Self {
        Self::with_seed(tournament, size, scoring_config, DEFAULT_POOL_SEED)
    }

    pub fn with_seed(
        tournament: &TournamentInfo,
        size: usize,
        scoring_config: &ScoringConfig,
        seed: u64,
    ) -> Self {
        MonteCarloScenarios {
            pool: ScenarioPool::new(tournament, size, scoring_config, seed),
            size,
            score_table: ScoreTable::new(scoring_config),
        }
    }

    /// Silent constructor plus a one-line note, so the progress chatter lives
    /// with the caller that wanted it rather than inside the data structure.
    pub fn announced(
        tournament: &TournamentInfo,
        size: usize,
        scoring_config: &ScoringConfig,
    ) -> Self {
        println!("Generating simulation pool of {} scenarios...", size);
        let pool = Self::new(tournament, size, scoring_config);
        println!("Simulation pool generated.");
        pool
    }

    fn prepare(&self, bracket: &Bracket) -> ScoredPicks {
        ScoredPicks::from_winners(&bracket.winner_indices(), self.pool.points())
    }

    /// Mean score of a single bracket across the pool.
    pub fn score_bracket(&self, bracket: &Bracket, _scoring_config: &ScoringConfig) -> f64 {
        self.pool.par_mean_score(&self.prepare(bracket))
    }

    /// Best-ball score of a portfolio: the mean over scenarios of the best
    /// entry's score in that scenario.
    pub fn score_portfolio_best_ball(
        &self,
        portfolio: &[Bracket],
        _scoring_config: &ScoringConfig,
    ) -> f64 {
        if portfolio.is_empty() {
            return 0.0;
        }
        let prepared: Vec<ScoredPicks> = portfolio.iter().map(|b| self.prepare(b)).collect();
        self.pool.par_best_ball_mean(&prepared)
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
    /// Draw a mutation: a team, and how far up the bracket to push it.
    ///
    /// Earlier rounds are weighted more heavily because they move more picks —
    /// forcing a team into the round of 32 rewrites one game, forcing it to the
    /// title rewrites six.
    pub fn random_move(tournament: &TournamentInfo, rng: &mut impl Rng) -> (u8, usize) {
        let team = rng.gen_range(0..tournament.teams.len()) as u8;
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
        (team, wins)
    }

    /// Apply a random team-round move to `picks`, in place.
    pub fn mutate_picks(picks: &mut Picks, tournament: &TournamentInfo, rng: &mut impl Rng) {
        let (team, wins) = Self::random_move(tournament, rng);
        picks.force_to_round(tournament, team, wins);
    }

    /// Pick a random team and force it to win a random number of games.
    pub fn mutate(
        bracket: &Bracket,
        tournament: &TournamentInfo,
        scoring_config: &ScoringConfig,
    ) -> Bracket {
        let mut rng = rand::thread_rng();
        let mut picks = bracket.picks(tournament);
        Self::mutate_picks(&mut picks, tournament, &mut rng);
        Bracket::from_picks(tournament, &picks, Some(scoring_config))
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
    /// Kept for callers that still hold a `Vec<bool>`; the work happens in
    /// [`Picks::force_to_round`], which walks the twelve games the move can
    /// reach instead of re-decoding all sixty-three.
    pub fn force_index_to_round(
        binary: &[bool],
        tournament: &TournamentInfo,
        team_index: u8,
        wins: usize,
    ) -> Vec<bool> {
        let mut bits = 0u64;
        for (game, &bit) in binary.iter().enumerate() {
            if bit {
                bits |= 1u64 << game;
            }
        }
        let mut picks = Picks::from_bits(tournament, bits);
        picks.force_to_round(tournament, team_index, wins);
        (0..binary.len()).map(|g| picks.bit(g)).collect()
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

    /// Force every lock back into a candidate, in place.
    ///
    /// This runs on every candidate the optimizers produce, so it checks first
    /// and rewrites only when something has actually drifted.
    #[inline]
    pub fn repair_picks(&self, picks: &mut Picks, tournament: &TournamentInfo) {
        if self.locks.is_empty() || self.holds(picks) {
            return;
        }
        for lock in &self.locks {
            picks.force_to_round(tournament, lock.team_index, lock.wins_required);
        }
    }

    /// Whether every lock already holds in `picks`.
    #[inline]
    pub fn holds(&self, picks: &Picks) -> bool {
        self.locks
            .iter()
            .all(|lock| picks.wins_for(lock.team_index) >= lock.wins_required)
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

/// Individual in the GA population.
///
/// Carries the compact [`Picks`] rather than a full `Bracket`: a generation of
/// 100 individuals used to mean 6,300 `Game` structs and ~19,000 atomic
/// refcount operations per generation, all of it thrown away at the next.
#[derive(Clone, Copy)]
pub struct Individual {
    pub picks: Picks,
    pub fitness: f64,
    /// Cleared when the candidate changes; lets the evaluator skip elites and
    /// straight clones, which are typically a quarter of every generation.
    evaluated: bool,
}

impl Individual {
    pub fn new(picks: Picks) -> Self {
        Individual {
            picks,
            fitness: f64::NEG_INFINITY,
            evaluated: false,
        }
    }

    pub fn bracket(&self, tournament: &TournamentInfo, scoring: &ScoringConfig) -> Bracket {
        Bracket::from_picks(tournament, &self.picks, Some(scoring))
    }
}

/// Seed a population of random brackets.
fn random_population(
    tournament: &TournamentInfo,
    size: usize,
    seed: u64,
) -> Vec<Picks> {
    // Chunked so the work parallelizes while staying reproducible for a seed.
    const CHUNK: usize = 64;
    let chunks = size.div_ceil(CHUNK);
    let mut out: Vec<Vec<Picks>> = (0..chunks)
        .into_par_iter()
        .map(|c| {
            let mut rng = SmallRng::seed_from_u64(seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let n = CHUNK.min(size - c * CHUNK);
            (0..n).map(|_| Picks::sample(tournament, &mut rng)).collect()
        })
        .collect();
    let mut flat = Vec::with_capacity(size);
    for chunk in out.drain(..) {
        flat.extend(chunk);
    }
    flat
}

/// Team-round crossover: take the base parent's bracket and inject a handful of
/// (team, round) outcomes drawn from the donor.
///
/// The genes are advancement facts — "this team reached the Sweet 16" — rather
/// than raw bits, because a bit's meaning depends on who reaches its game.
fn team_round_crossover(
    base: &Picks,
    donor: &Picks,
    tournament: &TournamentInfo,
    rng: &mut impl Rng,
) -> Picks {
    let mut child = *base;
    for _ in 0..rng.gen_range(1..=4) {
        let round = rng.gen_range(1..=6usize);
        let (start, count) = ROUND_SLOTS[round - 1];
        let team = donor.winner(start + rng.gen_range(0..count));
        child.force_to_round(tournament, team, round);
    }
    child
}

/// (first game index, game count) of each round, indexed by round - 1.
const ROUND_SLOTS: [(usize, usize); 6] = [(0, 32), (32, 16), (48, 8), (56, 4), (60, 2), (62, 1)];

/// Genetic Algorithm for bracket optimization
pub struct GeneticAlgorithm {
    pub population: Vec<Individual>,
    pub settings: GaSettings,
    pub scoring_config: ScoringConfig,
    pub generation: usize,
    pub best_fitness: f64,
    pub best_picks: Option<Picks>,
    /// Re-applied to every candidate; see `LockSet`.
    pub locks: LockSet,
    rng: SmallRng,
}

impl GeneticAlgorithm {
    /// Create a new GA with random initial population
    pub fn new(
        tournament: &TournamentInfo,
        settings: GaSettings,
        scoring_config: ScoringConfig,
    ) -> Self {
        Self::with_seed(tournament, settings, scoring_config, rand::random())
    }

    /// As `new`, from a fixed seed — the whole search becomes reproducible.
    pub fn with_seed(
        tournament: &TournamentInfo,
        settings: GaSettings,
        scoring_config: ScoringConfig,
        seed: u64,
    ) -> Self {
        let population = random_population(tournament, settings.population_size, seed)
            .into_iter()
            .map(Individual::new)
            .collect();

        GeneticAlgorithm {
            population,
            settings,
            scoring_config,
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
            best_picks: None,
            locks: LockSet::default(),
            rng: SmallRng::seed_from_u64(seed ^ 0x5DEE_CE66_D000_0000),
        }
    }

    /// Constrain the search: every candidate, starting with the initial
    /// population, is repaired to satisfy these locks.
    pub fn with_locks(mut self, locks: LockSet, tournament: &TournamentInfo) -> Self {
        for ind in self.population.iter_mut() {
            locks.repair_picks(&mut ind.picks, tournament);
        }
        self.locks = locks;
        self
    }

    /// The best bracket found so far, materialised for display.
    pub fn best_bracket(&self, tournament: &TournamentInfo) -> Option<Bracket> {
        self.best_picks
            .map(|p| Bracket::from_picks(tournament, &p, Some(&self.scoring_config)))
    }

    /// Score every candidate against `baseline`, the per-scenario best of the
    /// already-frozen part of the portfolio.
    ///
    /// An empty baseline (all zeros) makes this plain expected score, so the
    /// single-bracket and portfolio cases run the same code. Parallelism is one
    /// level only — across the population, with the scenario loop serial inside
    /// — because rayon nested in rayon spent more time splitting jobs than
    /// scoring: it was a quarter of total run time in the profile.
    fn evaluate(&mut self, pool: &ScenarioPool, baseline: Option<&[f32]>) {
        let points = pool.points();
        let updates: Vec<(usize, f64)> = self
            .population
            .par_iter()
            .enumerate()
            .filter(|(_, ind)| !ind.evaluated)
            .map(|(i, ind)| {
                let prepared = ScoredPicks::new(&ind.picks, points);
                let fitness = match baseline {
                    Some(base) => pool.mean_max_with(&prepared, base),
                    None => pool.mean_score(&prepared),
                };
                (i, fitness)
            })
            .collect();

        for (i, fitness) in updates {
            self.population[i].fitness = fitness;
            self.population[i].evaluated = true;
        }

        if let Some(best) = self
            .population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        {
            if best.fitness > self.best_fitness {
                self.best_fitness = best.fitness;
                self.best_picks = Some(best.picks);
            }
        }
    }

    /// Evaluate fitness for all individuals using the simulation pool.
    pub fn evaluate_fitness(&mut self, pool: &MonteCarloScenarios) {
        self.evaluate(&pool.pool, None);
    }

    /// Tournament selection - pick best from random subset
    fn select(&self, rng: &mut impl Rng) -> usize {
        let mut best = rng.gen_range(0..self.population.len());
        for _ in 1..self.settings.tournament_size {
            let challenger = rng.gen_range(0..self.population.len());
            if self.population[challenger].fitness > self.population[best].fitness {
                best = challenger;
            }
        }
        best
    }

    /// Run one generation of evolution
    pub fn evolve_generation(&mut self, tournament: &TournamentInfo) {
        let mut rng = std::mem::replace(&mut self.rng, SmallRng::seed_from_u64(0));
        let elites = self.settings.elitism_count.min(self.population.len());

        // Partial sort: only the elite prefix has to be in order, and the
        // survivors are moved rather than cloned.
        self.population
            .sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut next: Vec<Individual> = Vec::with_capacity(self.settings.population_size);
        next.extend_from_slice(&self.population[..elites]);

        while next.len() < self.settings.population_size {
            let p1 = self.select(&mut rng);
            let p2 = self.select(&mut rng);

            let crossed = rng.gen::<f64>() < self.settings.crossover_rate;
            let mutated = rng.gen::<f64>() < self.settings.mutation_rate;

            let mut child = if crossed {
                let (base, donor) = if rng.gen::<bool>() { (p1, p2) } else { (p2, p1) };
                team_round_crossover(
                    &self.population[base].picks,
                    &self.population[donor].picks,
                    tournament,
                    &mut rng,
                )
            } else {
                self.population[p1].picks
            };

            if mutated {
                TeamRoundMutator::mutate_picks(&mut child, tournament, &mut rng);
            }
            self.locks.repair_picks(&mut child, tournament);

            let mut individual = Individual::new(child);
            // An untouched clone of an already-scored parent keeps its fitness.
            if !crossed && !mutated && child == self.population[p1].picks {
                individual.fitness = self.population[p1].fitness;
                individual.evaluated = self.population[p1].evaluated;
            }
            next.push(individual);
        }

        self.population = next;
        self.generation += 1;
        self.rng = rng;
    }

    /// Run the full GA optimization
    pub fn run(
        &mut self,
        tournament: &TournamentInfo,
        pool: &MonteCarloScenarios,
        verbose: bool,
    ) -> Bracket {
        self.run_against(tournament, &pool.pool, None, verbose, "Best fitness");
        self.result(tournament)
    }

    /// Run GA for portfolio mode, maximizing best-ball score once this bracket
    /// joins the portfolio whose per-scenario best is `baseline`.
    pub fn run_for_portfolio(
        &mut self,
        tournament: &TournamentInfo,
        pool: &MonteCarloScenarios,
        baseline: Option<&[f32]>,
        verbose: bool,
    ) -> Bracket {
        let label = if baseline.is_some() {
            "Portfolio best-ball"
        } else {
            "Expected score"
        };
        self.run_against(tournament, &pool.pool, baseline, verbose, label);
        self.result(tournament)
    }

    fn run_against(
        &mut self,
        tournament: &TournamentInfo,
        pool: &ScenarioPool,
        baseline: Option<&[f32]>,
        verbose: bool,
        label: &str,
    ) {
        for gen in 0..self.settings.generations {
            self.evaluate(pool, baseline);
            if verbose && gen % 20 == 0 {
                let avg = self.population.iter().map(|i| i.fitness).sum::<f64>()
                    / self.population.len() as f64;
                println!(
                    "Generation {}: {} = {:.2}, Avg = {:.2}",
                    gen, label, self.best_fitness, avg
                );
            }
            self.evolve_generation(tournament);
        }
        self.evaluate(pool, baseline);
        if verbose {
            println!("Final: {} = {:.2}", label, self.best_fitness);
        }
    }

    fn result(&self, tournament: &TournamentInfo) -> Bracket {
        let picks = self.best_picks.unwrap_or(self.population[0].picks);
        Bracket::from_picks(tournament, &picks, Some(&self.scoring_config))
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
    pub picks: Vec<Picks>,
    pub fitness: f64,
    evaluated: bool,
}

impl PortfolioIndividual {
    pub fn new(picks: Vec<Picks>) -> Self {
        PortfolioIndividual {
            picks,
            fitness: f64::NEG_INFINITY,
            evaluated: false,
        }
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
    pub best_portfolio: Option<Vec<Picks>>,
    /// Applied to every bracket of every portfolio.
    pub locks: LockSet,
    rng: SmallRng,
}

impl WholePortfolioGA {
    pub fn new(
        tournament: &TournamentInfo,
        num_brackets: usize,
        settings: GaSettings,
        scoring_config: ScoringConfig,
    ) -> Self {
        Self::with_seed(
            tournament,
            num_brackets,
            settings,
            scoring_config,
            rand::random(),
        )
    }

    pub fn with_seed(
        tournament: &TournamentInfo,
        num_brackets: usize,
        settings: GaSettings,
        scoring_config: ScoringConfig,
        seed: u64,
    ) -> Self {
        let flat = random_population(
            tournament,
            settings.population_size * num_brackets.max(1),
            seed,
        );
        let population = flat
            .chunks(num_brackets.max(1))
            .map(|c: &[Picks]| PortfolioIndividual::new(c.to_vec()))
            .collect();

        WholePortfolioGA {
            population,
            settings,
            scoring_config,
            num_brackets,
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
            best_portfolio: None,
            locks: LockSet::default(),
            rng: SmallRng::seed_from_u64(seed ^ 0x1234_5678_9ABC_DEF0),
        }
    }

    /// Constrain every bracket in every portfolio to satisfy these locks.
    pub fn with_locks(mut self, locks: LockSet, tournament: &TournamentInfo) -> Self {
        for ind in self.population.iter_mut() {
            for picks in ind.picks.iter_mut() {
                locks.repair_picks(picks, tournament);
            }
        }
        self.locks = locks;
        self
    }

    /// Evaluate fitness for all portfolios using best-ball scoring
    pub fn evaluate_fitness(&mut self, pool: &MonteCarloScenarios) {
        let scenarios = &pool.pool;
        let points = scenarios.points();
        let updates: Vec<(usize, f64)> = self
            .population
            .par_iter()
            .enumerate()
            .filter(|(_, ind)| !ind.evaluated)
            .map(|(i, ind)| {
                let prepared: Vec<ScoredPicks> = ind
                    .picks
                    .iter()
                    .map(|p| ScoredPicks::new(p, points))
                    .collect();
                (i, scenarios.best_ball_mean(&prepared))
            })
            .collect();

        for (i, fitness) in updates {
            self.population[i].fitness = fitness;
            self.population[i].evaluated = true;
        }

        if let Some(best) = self
            .population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        {
            if best.fitness > self.best_fitness {
                self.best_fitness = best.fitness;
                self.best_portfolio = Some(best.picks.clone());
            }
        }
    }

    fn select(&self, rng: &mut impl Rng) -> usize {
        let mut best = rng.gen_range(0..self.population.len());
        for _ in 1..self.settings.tournament_size {
            let challenger = rng.gen_range(0..self.population.len());
            if self.population[challenger].fitness > self.population[best].fitness {
                best = challenger;
            }
        }
        best
    }

    /// Run one generation of evolution
    pub fn evolve_generation(&mut self, tournament: &TournamentInfo) {
        let mut rng = std::mem::replace(&mut self.rng, SmallRng::seed_from_u64(0));
        let elites = self.settings.elitism_count.min(self.population.len());

        self.population
            .sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut next: Vec<PortfolioIndividual> =
            Vec::with_capacity(self.settings.population_size);
        next.extend_from_slice(&self.population[..elites]);

        while next.len() < self.settings.population_size {
            let p1 = self.select(&mut rng);
            let p2 = self.select(&mut rng);

            let crossed = rng.gen::<f64>() < self.settings.crossover_rate;
            let mutated = rng.gen::<f64>() < self.settings.mutation_rate;

            let mut child: Vec<Picks> = if crossed {
                (0..self.population[p1].picks.len())
                    .map(|i| {
                        let (base, donor) =
                            if rng.gen::<bool>() { (p1, p2) } else { (p2, p1) };
                        team_round_crossover(
                            &self.population[base].picks[i],
                            &self.population[donor].picks[i],
                            tournament,
                            &mut rng,
                        )
                    })
                    .collect()
            } else {
                self.population[p1].picks.clone()
            };

            if mutated && !child.is_empty() {
                let idx = rng.gen_range(0..child.len());
                TeamRoundMutator::mutate_picks(&mut child[idx], tournament, &mut rng);
            }
            for picks in child.iter_mut() {
                self.locks.repair_picks(picks, tournament);
            }

            let unchanged = !crossed && !mutated && child == self.population[p1].picks;
            let mut individual = PortfolioIndividual::new(child);
            if unchanged {
                individual.fitness = self.population[p1].fitness;
                individual.evaluated = self.population[p1].evaluated;
            }
            next.push(individual);
        }

        self.population = next;
        self.generation += 1;
        self.rng = rng;
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

        self.evaluate_fitness(pool);

        if verbose {
            println!("Final: Best fitness = {:.2}", self.best_fitness);
        }

        let picks = self
            .best_portfolio
            .clone()
            .unwrap_or_else(|| self.population[0].picks.clone());
        picks
            .iter()
            .map(|p| Bracket::from_picks(tournament, p, Some(&self.scoring_config)))
            .collect()
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

    /// Optimize a portfolio of brackets sequentially.
    ///
    /// Each frozen bracket is folded into a running per-scenario best, so the
    /// `n`-th bracket is still evaluated with one bracket's worth of scoring
    /// rather than `n`. That baseline used to be rebuilt from scratch for every
    /// candidate of every generation — for a five-bracket portfolio, four
    /// fifths of the work was recomputing something that had not changed.
    pub fn optimize(
        &self,
        tournament: &TournamentInfo,
        num_brackets: usize,
        verbose: bool,
    ) -> Vec<Bracket> {
        let scenarios = MonteCarloScenarios::announced(
            tournament,
            self.config.simulation.pool_size,
            &self.scoring_config,
        );
        let pool = &scenarios.pool;

        let mut portfolio: Vec<Picks> = Vec::with_capacity(num_brackets);
        // Per-scenario best over everything frozen so far.
        let mut baseline = vec![0.0f32; pool.size()];

        for i in 0..num_brackets {
            println!("\n=== Optimizing Bracket {} of {} ===", i + 1, num_brackets);

            let mut ga = GeneticAlgorithm::new(
                tournament,
                self.config.ga.clone(),
                self.scoring_config,
            )
            .with_locks(self.locks.clone(), tournament);

            let carry = if i == 0 { None } else { Some(&baseline[..]) };
            ga.run_for_portfolio(tournament, &scenarios, carry, verbose);

            let picks = ga.best_picks.unwrap_or(ga.population[0].picks);
            let prepared = pool.prepare(&picks);
            pool.absorb_into(&prepared, &mut baseline);

            let bracket = Bracket::from_picks(tournament, &picks, Some(&self.scoring_config));
            println!(
                "Bracket {}: Champion = {} (seed {}), EV = {:.2}",
                i + 1,
                bracket.winner.name,
                bracket.winner.seed,
                bracket.expected_value
            );
            println!(
                "Portfolio best-ball score after bracket {}: {:.2}",
                i + 1,
                mean(&baseline)
            );

            portfolio.push(picks);
        }

        println!("\n=== Portfolio Optimization Complete ===");
        println!("Final portfolio best-ball score: {:.2}", mean(&baseline));

        println!("\nIndividual bracket scores:");
        let brackets: Vec<Bracket> = portfolio
            .iter()
            .map(|p| Bracket::from_picks(tournament, p, Some(&self.scoring_config)))
            .collect();
        for (i, (picks, bracket)) in portfolio.iter().zip(brackets.iter()).enumerate() {
            println!(
                "  Bracket {}: {} - Score: {:.2}, EV: {:.2}",
                i + 1,
                bracket.winner.name,
                pool.mean_score(&pool.prepare(picks)),
                bracket.expected_value
            );
        }

        brackets
    }
}

/// Mean of a per-scenario profile.
fn mean(profile: &[f32]) -> f64 {
    profile.iter().map(|&v| v as f64).sum::<f64>() / profile.len() as f64
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
    pub fn optimize_single(&self, tournament: &TournamentInfo, verbose: bool) -> Bracket {
        let scenarios = MonteCarloScenarios::announced(
            tournament,
            self.config.simulation.pool_size,
            &self.scoring_config,
        );
        let pool = &scenarios.pool;

        let mut rng = SmallRng::from_entropy();

        let mut current = Picks::sample(tournament, &mut rng);
        self.locks.repair_picks(&mut current, tournament);
        let mut current_score = pool.par_mean_score(&pool.prepare(&current));

        let mut best = current;
        let mut best_score = current_score;

        let initial_temp: f64 = 10.0;
        let final_temp: f64 = 0.1;
        let cooling_rate =
            (final_temp / initial_temp).powf(1.0 / self.config.ga.generations as f64);
        let mut temperature = initial_temp;

        for gen in 0..self.config.ga.generations {
            let mut neighbor = current;
            TeamRoundMutator::mutate_picks(&mut neighbor, tournament, &mut rng);
            self.locks.repair_picks(&mut neighbor, tournament);

            let neighbor_score = pool.par_mean_score(&pool.prepare(&neighbor));

            let accept = if neighbor_score > current_score {
                true
            } else {
                let delta = neighbor_score - current_score;
                rng.gen::<f64>() < (delta / temperature).exp()
            };

            if accept {
                current = neighbor;
                current_score = neighbor_score;

                if current_score > best_score {
                    best = current;
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

        Bracket::from_picks(tournament, &best, Some(&self.scoring_config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bracket::tests::assert_legal;
    use crate::ingest::tests::tournament;
    use crate::tree::NO_GAME;

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
            let child = team_round_crossover(
                &a.picks(&t),
                &b.picks(&t),
                &t,
                &mut rng,
            );
            assert_legal(&Bracket::from_picks(&t, &child, Some(&scoring)));
        }
    }
}
