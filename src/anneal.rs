use rand::Rng;
use crate::bracket::Bracket;
use crate::ingest::TournamentInfo;
use crate::pool::Batch;

/// Schedule for simulated annealing
pub struct AnnealingSchedule {
    pub initial_temp: f64,
    pub cooling_rate: f64,
    pub min_temp: f64,
    pub steps_per_temp: usize,
}

impl AnnealingSchedule {
    pub fn new(steps_per_temp: u32) -> Self {
        AnnealingSchedule {
            initial_temp: 100.0,
            cooling_rate: 0.95,
            min_temp: 0.001,
            steps_per_temp: steps_per_temp as usize,
        }
    }
}

/// Optimize a bracket to maximize portfolio performance
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
    schedule: &AnnealingSchedule,
) -> Bracket {
    let mut current_bracket = Bracket::new(tournament);
    let mut current_fitness = calculate_portfolio_fitness(&current_bracket, existing_portfolio, simulation_pool);

    let mut best_bracket = current_bracket.clone();
    let mut best_fitness = current_fitness;

    let mut temp = schedule.initial_temp;
    let mut rng = rand::thread_rng();

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

    while temp > schedule.min_temp {
        for _ in 0..schedule.steps_per_temp {
            // Mutate
            // Use a small mutation rate (1-3 bits)
            let mutation_rate = 1.0 / 63.0 * 2.0;
            let neighbor = current_bracket.mutate(tournament, mutation_rate);

            // Calculate fitness efficiently
            let neighbor_fitness = calculate_fitness_with_precalc(&neighbor, &existing_max_scores, simulation_pool);

            let delta = neighbor_fitness - current_fitness;

            if delta > 0.0 {
                current_bracket = neighbor;
                current_fitness = neighbor_fitness;

                if current_fitness > best_fitness {
                    best_bracket = current_bracket.clone();
                    best_fitness = current_fitness;
                }
            } else {
                let probability = (delta / temp).exp();
                if rng.gen::<f64>() < probability {
                    current_bracket = neighbor;
                    current_fitness = neighbor_fitness;
                }
            }
        }
        temp *= schedule.cooling_rate;
    }

    best_bracket
}

/// Calculate fitness: Average of max(existing_best, candidate_score) across all simulations
fn calculate_portfolio_fitness(
    candidate: &Bracket,
    existing_portfolio: &[Bracket],
    pool: &Batch,
) -> f64 {
    let total_score: f64 = pool.brackets.iter().map(|sim| {
        let mut max_score = candidate.score(sim);
        for existing in existing_portfolio {
            let s = existing.score(sim);
            if s > max_score {
                max_score = s;
            }
        }
        max_score
    }).sum();

    total_score / pool.brackets.len() as f64
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
