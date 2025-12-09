use rand::Rng;
use crate::bracket::Bracket;
use crate::ingest::TournamentInfo;
use crate::pool::Batch;

/// Configuration for Genetic Algorithm
pub struct GeneticConfig {
    pub generations: u32,
    pub population_size: usize,
    pub elitism_count: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
}

impl GeneticConfig {
    pub fn new(generations: u32) -> Self {
        GeneticConfig {
            generations,
            population_size: 50, // Keep small for portfolio loop efficiency
            elitism_count: 5,
            crossover_rate: 0.7,
            mutation_rate: 1.0 / 63.0 * 2.0, // Base rate for smart mutation
        }
    }
}

/// Optimize a bracket to maximize portfolio performance using Genetic Algorithm
///
/// Fitness function: Average maximum score across the simulation pool
/// For each simulation S in pool:
///   Score = max(score(existing_1, S), ..., score(existing_N, S), score(candidate, S))
///
/// We maximize the average of these scores.
pub fn optimize_portfolio_bracket(
    tournament: &TournamentInfo,
    existing_portfolio: &[Bracket],
    simulation_pool: &Batch,
    config: &GeneticConfig,
) -> Bracket {
    let mut rng = rand::thread_rng();

    // Initialize population
    let mut population: Vec<Bracket> = Vec::with_capacity(config.population_size);
    for _ in 0..config.population_size {
        // Mix random brackets and mutated copies of the current best (if any)
        population.push(Bracket::new(tournament));
    }

    // Pre-calculate max scores for existing portfolio to avoid recomputing
    // For each simulation, we only need to know the max score of the existing brackets
    let existing_max_scores: Vec<f64> = if existing_portfolio.is_empty() {
        vec![0.0; simulation_pool.brackets.len()]
    } else {
        simulation_pool.brackets.iter().map(|sim| {
            existing_portfolio.iter()
                .map(|b| b.score(sim))
                .fold(0.0, f64::max)
        }).collect()
    };

    // Main evolution loop
    for _gen in 0..config.generations {
        // Calculate fitness for all individuals
        let mut fitness_scores: Vec<(usize, f64)> = population.iter().enumerate().map(|(idx, bracket)| {
            (idx, calculate_fitness_with_precalc(bracket, &existing_max_scores, simulation_pool))
        }).collect();

        // Sort by fitness descending
        fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Elitism: Keep the best individuals
        let mut new_population = Vec::with_capacity(config.population_size);
        for i in 0..config.elitism_count {
            if i < fitness_scores.len() {
                new_population.push(population[fitness_scores[i].0].clone());
            }
        }

        // Generate rest of new population
        while new_population.len() < config.population_size {
            // Selection (Tournament)
            let parent1 = tournament_selection(&population, &fitness_scores, &mut rng);
            let parent2 = tournament_selection(&population, &fitness_scores, &mut rng);

            // Crossover
            let mut child = if rng.gen::<f64>() < config.crossover_rate {
                parent1.crossover(parent2, tournament)
            } else {
                parent1.clone()
            };

            // Mutation (Smart Mutate)
            child = child.smart_mutate(tournament, config.mutation_rate);

            new_population.push(child);
        }

        population = new_population;
    }

    // Return the best individual from the final generation
    let best_idx = population.iter()
        .map(|b| calculate_fitness_with_precalc(b, &existing_max_scores, simulation_pool))
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    population[best_idx].clone()
}

fn tournament_selection<'a>(
    population: &'a [Bracket],
    fitness_scores: &[(usize, f64)],
    rng: &mut impl Rng,
) -> &'a Bracket {
    let k = 3;
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fitness = 0.0;

    // Find fitness of initial random pick (naive search)
    for (idx, score) in fitness_scores {
        if *idx == best_idx {
            best_fitness = *score;
            break;
        }
    }

    for _ in 1..k {
        let idx = rng.gen_range(0..population.len());
        let mut fitness = 0.0;
        for (f_idx, f_score) in fitness_scores {
            if *f_idx == idx {
                fitness = *f_score;
                break;
            }
        }

        if fitness > best_fitness {
            best_fitness = fitness;
            best_idx = idx;
        }
    }

    &population[best_idx]
}

/// Optimized fitness calculation using pre-calculated max scores of existing portfolio
fn calculate_fitness_with_precalc(
    candidate: &Bracket,
    existing_max_scores: &[f64],
    pool: &Batch,
) -> f64 {
    let total_score: f64 = pool.brackets.iter().zip(existing_max_scores.iter()).map(|(sim, &existing_max)| {
        let score = candidate.score(sim);
        if score > existing_max {
            score
        } else {
            existing_max
        }
    }).sum();

    total_score / pool.brackets.len() as f64
}
