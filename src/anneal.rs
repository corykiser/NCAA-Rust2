use crate::bracket::{Bracket, ScoringConfig};
use crate::ga::MonteCarloScenarios;
use crate::ingest::TournamentInfo;
use crate::portfolio::BracketPortfolio;
use rand::Rng;

pub struct AnnealingConfig {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub steps: usize,
    /// Pool size for Monte Carlo scoring
    pub pool_size: usize,
    /// Legacy: diversity weight (no longer used, kept for API compatibility)
    pub diversity_weight: f64,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 10.0,
            cooling_rate: 0.9995,
            steps: 10000,
            pool_size: 10000,
            diversity_weight: 0.0, // No longer used
        }
    }
}

/// Optimize a portfolio using Simulated Annealing with best-ball scoring
///
/// The fitness function is the average best-ball score across all simulations:
/// For each simulation, take max(score of each bracket), then average across simulations.
///
/// This naturally encourages diversity because brackets that cover different
/// tournament outcomes will improve the portfolio's best-ball score.
pub fn optimize_portfolio(
    tournament: &TournamentInfo,
    num_brackets: usize,
    config: AnnealingConfig,
    scoring_config: &ScoringConfig,
) -> BracketPortfolio {
    let mut rng = rand::thread_rng();

    println!("Generating simulation pool of {} brackets for scoring...", config.pool_size);
    let pool = MonteCarloScenarios::new(tournament, config.pool_size, scoring_config);

    // Initialize random portfolio
    let mut current_brackets: Vec<Bracket> = (0..num_brackets)
        .map(|_| Bracket::new(tournament, Some(scoring_config)))
        .collect();

    let mut current_score = pool.score_portfolio_best_ball(&current_brackets, scoring_config);

    let mut best_brackets = current_brackets.clone();
    let mut best_score = current_score;

    let mut temp = config.initial_temperature;

    println!("Starting Simulated Annealing optimization...");
    println!("Initial best-ball score: {:.2}", current_score);

    let report_interval = config.steps / 10;

    for step in 0..config.steps {
        // Create neighbor solution: mutate one random bracket
        let idx = rng.gen_range(0..num_brackets);
        let original_bracket = current_brackets[idx].clone();

        // Mutate with moderate rate
        let mutation_rate = 3.0 / 63.0;
        current_brackets[idx] = current_brackets[idx].mutate(tournament, mutation_rate, Some(scoring_config));

        let new_score = pool.score_portfolio_best_ball(&current_brackets, scoring_config);

        // Accept or reject based on Metropolis criterion
        let accept = if new_score > current_score {
            true
        } else {
            let delta = new_score - current_score;
            let acceptance_prob = (delta / temp).exp();
            rng.gen::<f64>() < acceptance_prob
        };

        if accept {
            current_score = new_score;
            if new_score > best_score {
                best_score = new_score;
                best_brackets = current_brackets.clone();
            }
        } else {
            // Revert the mutation
            current_brackets[idx] = original_bracket;
        }

        temp *= config.cooling_rate;

        // Progress reporting
        if report_interval > 0 && step % report_interval == 0 && step > 0 {
            println!(
                "Step {}/{}: Best = {:.2}, Current = {:.2}, Temp = {:.4}",
                step, config.steps, best_score, current_score, temp
            );
        }
    }

    println!("Optimization complete. Final best-ball score: {:.2}", best_score);

    // Print summary of champions
    println!("\nPortfolio champions:");
    for (i, bracket) in best_brackets.iter().enumerate() {
        println!(
            "  Bracket {}: {} (seed {})",
            i + 1,
            bracket.winner.name,
            bracket.winner.seed
        );
    }

    let mut portfolio = BracketPortfolio::new();
    portfolio.brackets = best_brackets;
    portfolio.constraints = vec![Vec::new(); num_brackets];

    portfolio
}

/// Optimize a portfolio using Simulated Annealing with smart mutations
/// Uses team/round based mutations instead of random bit flips
pub fn optimize_portfolio_smart(
    tournament: &TournamentInfo,
    num_brackets: usize,
    config: AnnealingConfig,
    scoring_config: &ScoringConfig,
) -> BracketPortfolio {
    use crate::ga::TeamRoundMutator;

    let mut rng = rand::thread_rng();

    println!("Generating simulation pool of {} brackets for scoring...", config.pool_size);
    let pool = MonteCarloScenarios::new(tournament, config.pool_size, scoring_config);

    // Initialize random portfolio
    let mut current_brackets: Vec<Bracket> = (0..num_brackets)
        .map(|_| Bracket::new(tournament, Some(scoring_config)))
        .collect();

    let mut current_score = pool.score_portfolio_best_ball(&current_brackets, scoring_config);

    let mut best_brackets = current_brackets.clone();
    let mut best_score = current_score;

    let mut temp = config.initial_temperature;

    println!("Starting Simulated Annealing with smart mutations...");
    println!("Initial best-ball score: {:.2}", current_score);

    let report_interval = config.steps / 10;

    for step in 0..config.steps {
        // Create neighbor solution: mutate one random bracket using smart mutation
        let idx = rng.gen_range(0..num_brackets);
        let original_bracket = current_brackets[idx].clone();

        // Use smart mutation (80% of time) or bit-flip (20%)
        current_brackets[idx] = if rng.gen::<f64>() < 0.8 {
            TeamRoundMutator::mutate(&current_brackets[idx], tournament, scoring_config)
        } else {
            current_brackets[idx].mutate(tournament, 3.0 / 63.0, Some(scoring_config))
        };

        let new_score = pool.score_portfolio_best_ball(&current_brackets, scoring_config);

        // Accept or reject
        let accept = if new_score > current_score {
            true
        } else {
            let delta = new_score - current_score;
            let acceptance_prob = (delta / temp).exp();
            rng.gen::<f64>() < acceptance_prob
        };

        if accept {
            current_score = new_score;
            if new_score > best_score {
                best_score = new_score;
                best_brackets = current_brackets.clone();
            }
        } else {
            current_brackets[idx] = original_bracket;
        }

        temp *= config.cooling_rate;

        if report_interval > 0 && step % report_interval == 0 && step > 0 {
            println!(
                "Step {}/{}: Best = {:.2}, Current = {:.2}, Temp = {:.4}",
                step, config.steps, best_score, current_score, temp
            );
        }
    }

    println!("Optimization complete. Final best-ball score: {:.2}", best_score);

    println!("\nPortfolio champions:");
    for (i, bracket) in best_brackets.iter().enumerate() {
        println!(
            "  Bracket {}: {} (seed {})",
            i + 1,
            bracket.winner.name,
            bracket.winner.seed
        );
    }

    let mut portfolio = BracketPortfolio::new();
    portfolio.brackets = best_brackets;
    portfolio.constraints = vec![Vec::new(); num_brackets];

    portfolio
}
