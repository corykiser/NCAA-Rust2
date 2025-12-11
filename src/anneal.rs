use crate::bracket::{Bracket, ScoringConfig};
use crate::ingest::TournamentInfo;
use crate::portfolio::BracketPortfolio;
use rand::Rng;

pub struct AnnealingConfig {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub steps: usize,
    pub diversity_weight: f64,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 10.0,
            cooling_rate: 0.995,
            steps: 10000,
            diversity_weight: 1.0,
        }
    }
}

pub fn optimize_portfolio(
    tournament: &TournamentInfo,
    num_brackets: usize,
    config: AnnealingConfig,
    scoring_config: &ScoringConfig,
) -> BracketPortfolio {
    let mut rng = rand::thread_rng();

    // Initialize random portfolio
    let mut current_brackets: Vec<Bracket> = (0..num_brackets)
        .map(|_| Bracket::new(tournament, Some(scoring_config))) // Random brackets
        .collect();

    let mut current_score = calculate_portfolio_score(&current_brackets, config.diversity_weight, scoring_config);

    let mut best_brackets = current_brackets.clone();
    let mut best_score = current_score;

    let mut temp = config.initial_temperature;

    // Progress bar removed as we don't want to add dependency on indicatif here directly
    // unless it is already used. main.rs uses it, but let's keep this clean.

    for _ in 0..config.steps {
        // Create neighbor solution: mutate one bracket
        let idx = rng.gen_range(0..num_brackets);
        let original_bracket = current_brackets[idx].clone();

        // Mutate
        // Using a small mutation rate to make local moves
        let mutation_rate = 1.0 / 63.0 * 3.0;
        current_brackets[idx] = current_brackets[idx].mutate(tournament, mutation_rate, Some(scoring_config));

        let new_score = calculate_portfolio_score(&current_brackets, config.diversity_weight, scoring_config);

        // Accept or reject
        if new_score > current_score {
            current_score = new_score;
            if new_score > best_score {
                best_score = new_score;
                best_brackets = current_brackets.clone();
            }
        } else {
            let delta = new_score - current_score; // Negative
            let acceptance_prob = (delta / temp).exp();
            if rng.gen::<f64>() < acceptance_prob {
                current_score = new_score;
            } else {
                // Revert
                current_brackets[idx] = original_bracket;
            }
        }

        temp *= config.cooling_rate;
    }

    let mut portfolio = BracketPortfolio::new();
    portfolio.brackets = best_brackets;
    // Fill constraints with empty vectors as we don't track them in SA yet
    portfolio.constraints = vec![Vec::new(); num_brackets];

    portfolio
}

fn calculate_portfolio_score(brackets: &[Bracket], diversity_weight: f64, scoring_config: &ScoringConfig) -> f64 {
    let total_ev: f64 = brackets.iter().map(|b| b.expected_value).sum();
    let avg_ev = total_ev / brackets.len() as f64;

    if brackets.len() <= 1 {
        return avg_ev;
    }

    // Calculate diversity metric
    // We use average pairwise distance.
    // weighted_distance returns total_distance (higher = more different)
    // and similarity (1.0 = identical).
    // We want to maximize distance.

    let mut total_dist = 0.0;
    let mut count = 0;
    for i in 0..brackets.len() {
        for j in (i+1)..brackets.len() {
            // weighted_distance returns a struct. We use total_distance.
            total_dist += brackets[i].weighted_distance(&brackets[j], Some(scoring_config)).total_distance;
            count += 1;
        }
    }

    let avg_dist = if count > 0 {
        total_dist / count as f64
    } else {
        0.0
    };

    // Objective: Maximize EV + Diversity
    // We might need to normalize distance to be comparable to EV.
    // EV is around ~100-200 maybe?
    // total_distance max is around 32*2*32... it can be large.
    // BracketDistance::similarity is 0.0 to 1.0.

    // Let's use similarity instead, and minimize it.
    // Or just use total_distance as is, assuming diversity_weight handles scaling.

    avg_ev + diversity_weight * avg_dist
}
