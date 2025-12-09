// Portfolio generation for diverse bracket strategies
// Supports constrained bracket building and champion-stratified portfolios

use crate::bracket::Bracket;
use crate::ingest::{Team, RcTeam, TournamentInfo};
use crate::pool::Batch;
use crate::genetic::{self, GeneticConfig};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GeneticMode {
    /// Optimize brackets one by one (Sequential)
    Sequential,
    /// Optimize all brackets simultaneously (Holistic)
    Simultaneous,
}
use clap::ValueEnum;

/// Specifies how far a team must advance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdvancementRound {
    Round2,      // Must win at least 1 game
    Sweet16,     // Must reach Sweet 16
    Elite8,      // Must reach Elite 8
    FinalFour,   // Must reach Final Four
    Championship, // Must reach championship game
    Winner,      // Must win it all
}

impl AdvancementRound {
    /// Number of wins required to reach this round
    pub fn wins_required(&self) -> usize {
        match self {
            AdvancementRound::Round2 => 1,
            AdvancementRound::Sweet16 => 2,
            AdvancementRound::Elite8 => 3,
            AdvancementRound::FinalFour => 4,
            AdvancementRound::Championship => 5,
            AdvancementRound::Winner => 6,
        }
    }
}

/// A constraint on bracket generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BracketConstraint {
    pub team_name: String,
    pub must_reach: AdvancementRound,
}

impl BracketConstraint {
    pub fn new(team_name: &str, must_reach: AdvancementRound) -> Self {
        BracketConstraint {
            team_name: team_name.to_string(),
            must_reach,
        }
    }

    /// Shorthand for requiring a team to win the championship
    pub fn champion(team_name: &str) -> Self {
        Self::new(team_name, AdvancementRound::Winner)
    }

    /// Shorthand for requiring a team to reach the Final Four
    pub fn final_four(team_name: &str) -> Self {
        Self::new(team_name, AdvancementRound::FinalFour)
    }
}

/// Builds brackets with constraints
pub struct ConstrainedBracketBuilder<'a> {
    tournament: &'a TournamentInfo,
    constraints: Vec<BracketConstraint>,
}

impl<'a> ConstrainedBracketBuilder<'a> {
    pub fn new(tournament: &'a TournamentInfo) -> Self {
        ConstrainedBracketBuilder {
            tournament,
            constraints: Vec::new(),
        }
    }

    pub fn with_constraint(mut self, constraint: BracketConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn with_champion(self, team_name: &str) -> Self {
        self.with_constraint(BracketConstraint::champion(team_name))
    }

    pub fn with_final_four(self, team_name: &str) -> Self {
        self.with_constraint(BracketConstraint::final_four(team_name))
    }

    /// Find a team by name in the tournament
    /// Returns a clone of the RcTeam (cheap - just increments ref count)
    fn find_team(&self, name: &str) -> Option<RcTeam> {
        let name_lower = name.to_lowercase();
        self.tournament.teams.iter().find(|t| {
            t.name.to_lowercase() == name_lower ||
            t.name.to_lowercase().contains(&name_lower)
        }).map(Arc::clone)
    }

    /// Get the region index for a team
    fn get_region_index(&self, team: &Team) -> usize {
        match team.region.as_str() {
            "East" => 0,
            "West" => 1,
            "South" => 2,
            "Midwest" => 3,
            _ => 0,
        }
    }

    /// Build a bracket respecting all constraints
    /// Constrained games are set deterministically, others use probability
    pub fn build(&self) -> Result<Bracket, String> {
        // First, generate a random bracket as base
        let mut bracket = Bracket::new(self.tournament);

        // Apply each constraint
        for constraint in &self.constraints {
            let team = self.find_team(&constraint.team_name)
                .ok_or_else(|| format!("Team '{}' not found", constraint.team_name))?;

            self.apply_constraint(&mut bracket, &team, &constraint.must_reach)?;
        }

        // Recalculate bracket stats
        self.recalculate_bracket_stats(&mut bracket);

        Ok(bracket)
    }

    /// Build bracket optimized for EV while respecting constraints
    /// Uses expected value to decide unconstrained games
    pub fn build_optimal(&self) -> Result<Bracket, String> {
        // Start with a fresh bracket using probability-based picks
        let mut binary = self.generate_optimal_binary()?;

        // Apply constraints to the binary representation
        for constraint in &self.constraints {
            let team = self.find_team(&constraint.team_name)
                .ok_or_else(|| format!("Team '{}' not found", constraint.team_name))?;

            self.apply_constraint_to_binary(&mut binary, &team, &constraint.must_reach)?;
        }

        // Build bracket from binary (pass slice instead of Vec)
        let bracket = Bracket::new_from_binary(self.tournament, &binary);
        Ok(bracket)
    }

    /// Generate binary representation that maximizes expected value
    fn generate_optimal_binary(&self) -> Result<Vec<bool>, String> {
        let mut binary = Vec::with_capacity(63);

        // For each game, pick the higher probability winner
        // Round 1: 32 games - use O(1) lookups
        for region in &["East", "West", "South", "Midwest"] {
            for matchup in self.tournament.round1 {
                let team1 = self.tournament.get_team(region, matchup[0]);
                let team2 = self.tournament.get_team(region, matchup[1]);

                // Pick higher rated team (lower seed usually)
                let pick_first = team1.rating >= team2.rating;
                // hilo = true means lower seed wins
                let lower_seed_first = team1.seed < team2.seed;
                binary.push(pick_first == lower_seed_first);
            }
        }

        // For later rounds, we need to simulate forward
        // This is a simplification - just pick favorites
        // Round 2-6: add remaining bits based on probability
        for _ in 32..63 {
            binary.push(true); // Favor lower seeds / earlier alphabet
        }

        Ok(binary)
    }

    /// Apply a constraint to a bracket
    fn apply_constraint(
        &self,
        bracket: &mut Bracket,
        team: &RcTeam,
        must_reach: &AdvancementRound,
    ) -> Result<(), String> {
        let wins_needed = must_reach.wins_required();
        let region_idx = self.get_region_index(team);

        // Round 1: Ensure team wins their first game
        if wins_needed >= 1 {
            let r1_idx = self.find_round1_game_index(team, region_idx);
            if let Some(idx) = r1_idx {
                bracket.games[idx].winner = Arc::clone(team);
            }
        }

        // Round 2: Ensure team wins
        if wins_needed >= 2 {
            let r2_idx = self.find_round2_game_index(team, region_idx);
            if let Some(idx) = r2_idx {
                bracket.games[32 + idx].winner = Arc::clone(team);
            }
        }

        // Sweet 16 (Round 3)
        if wins_needed >= 3 {
            let r3_idx = self.find_round3_game_index(team, region_idx);
            if let Some(idx) = r3_idx {
                bracket.games[48 + idx].winner = Arc::clone(team);
            }
        }

        // Elite 8 (Round 4)
        if wins_needed >= 4 {
            let r4_idx = region_idx; // One Elite 8 game per region
            bracket.games[56 + r4_idx].winner = Arc::clone(team);
        }

        // Final Four (Round 5)
        if wins_needed >= 5 {
            // Final Four: South/Midwest play each other, East/West play each other
            let r5_idx = if region_idx == 2 || region_idx == 3 { 0 } else { 1 };
            bracket.games[60 + r5_idx].winner = Arc::clone(team);
        }

        // Championship (Round 6)
        if wins_needed >= 6 {
            bracket.games[62].winner = Arc::clone(team);
            bracket.winner = Arc::clone(team);
        }

        Ok(())
    }

    /// Apply constraint to binary representation
    fn apply_constraint_to_binary(
        &self,
        binary: &mut Vec<bool>,
        team: &RcTeam,
        must_reach: &AdvancementRound,
    ) -> Result<(), String> {
        let wins_needed = must_reach.wins_required();
        let region_idx = self.get_region_index(team);

        // Calculate binary indices for this team's games
        // Round 1: 8 games per region, starting at region_idx * 8
        if wins_needed >= 1 {
            let base = region_idx * 8;
            let game_in_region = self.seed_to_round1_game(team.seed);
            let idx = base + game_in_region;
            if idx < 32 {
                // Set to make team win (depends on seed position)
                binary[idx] = team.seed < 9; // Lower seeds are "true" in hilo
            }
        }

        // For later rounds, the logic is more complex because game indices
        // depend on who won earlier. We'll handle this by rebuilding the bracket
        // and then extracting the binary representation.
        // This is a limitation of the current approach.

        Ok(())
    }

    /// Find the Round 1 game index for a team
    fn find_round1_game_index(&self, team: &Team, region_idx: usize) -> Option<usize> {
        let base = region_idx * 8;
        let game_in_region = self.seed_to_round1_game(team.seed);
        Some(base + game_in_region)
    }

    /// Map seed to round 1 game within region (0-7)
    fn seed_to_round1_game(&self, seed: i32) -> usize {
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

    /// Find Round 2 game index
    fn find_round2_game_index(&self, team: &Team, region_idx: usize) -> Option<usize> {
        let base = region_idx * 4;
        let game_in_region = match team.seed {
            1 | 16 | 8 | 9 => 0,
            5 | 12 | 4 | 13 => 1,
            6 | 11 | 3 | 14 => 2,
            7 | 10 | 2 | 15 => 3,
            _ => 0,
        };
        Some(base + game_in_region)
    }

    /// Find Round 3 (Sweet 16) game index
    fn find_round3_game_index(&self, team: &Team, region_idx: usize) -> Option<usize> {
        let base = region_idx * 2;
        let game_in_region = match team.seed {
            1 | 16 | 8 | 9 | 5 | 12 | 4 | 13 => 0,
            6 | 11 | 3 | 14 | 7 | 10 | 2 | 15 => 1,
            _ => 0,
        };
        Some(base + game_in_region)
    }

    /// Recalculate bracket statistics after modifications
    fn recalculate_bracket_stats(&self, bracket: &mut Bracket) {
        let mut prob = 1.0;
        let mut score = 0.0;
        let mut expected_value = 0.0;

        // Round 1
        for game in &bracket.games[0..32] {
            prob *= game.winnerprob;
            score += 1.0 + game.winner.seed as f64;
            expected_value += game.winnerprob * (1.0 + game.winner.seed as f64);
        }

        // Round 2
        for game in &bracket.games[32..48] {
            prob *= game.winnerprob;
            score += 2.0 + game.winner.seed as f64;
            expected_value += game.winnerprob * (2.0 + game.winner.seed as f64);
        }

        // Round 3
        for game in &bracket.games[48..56] {
            prob *= game.winnerprob;
            score += 4.0 + game.winner.seed as f64;
            expected_value += game.winnerprob * (4.0 + game.winner.seed as f64);
        }

        // Round 4
        for game in &bracket.games[56..60] {
            prob *= game.winnerprob;
            score += 8.0 * game.winner.seed as f64;
            expected_value += game.winnerprob * (8.0 * game.winner.seed as f64);
        }

        // Round 5
        for game in &bracket.games[60..62] {
            prob *= game.winnerprob;
            score += 16.0 * game.winner.seed as f64;
            expected_value += game.winnerprob * (16.0 * game.winner.seed as f64);
        }

        // Round 6
        for game in &bracket.games[62..63] {
            prob *= game.winnerprob;
            score += 32.0 * game.winner.seed as f64;
            expected_value += game.winnerprob * (32.0 * game.winner.seed as f64);
        }

        bracket.prob = prob;
        bracket.score = score;
        bracket.expected_value = expected_value;
    }
}

/// A portfolio of diverse brackets
#[derive(Debug, Clone)]
pub struct BracketPortfolio {
    pub brackets: Vec<Bracket>,
    pub constraints: Vec<Vec<BracketConstraint>>,
}

impl BracketPortfolio {
    pub fn new() -> Self {
        BracketPortfolio {
            brackets: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Generate portfolio stratified by champion
    /// Each bracket bets on a different championship winner
    pub fn generate_champion_stratified(
        tournament: &TournamentInfo,
        num_brackets: usize,
    ) -> Self {
        let mut portfolio = BracketPortfolio::new();

        // Rank teams by rating (proxy for championship probability)
        // RcTeam dereferences to Team, so we can access .rating directly
        let mut ranked_teams: Vec<&RcTeam> = tournament.teams.iter().collect();
        ranked_teams.sort_by(|a, b| b.rating.partial_cmp(&a.rating).unwrap());

        // Generate one bracket per top team
        for team in ranked_teams.iter().take(num_brackets) {
            let constraint = BracketConstraint::champion(&team.name);

            let builder = ConstrainedBracketBuilder::new(tournament)
                .with_constraint(constraint.clone());

            match builder.build() {
                Ok(bracket) => {
                    portfolio.brackets.push(bracket);
                    portfolio.constraints.push(vec![constraint]);
                }
                Err(e) => {
                    eprintln!("Warning: Could not build bracket for {}: {}", team.name, e);
                }
            }
        }

        portfolio
    }

    /// Generate portfolio with greedy diversity
    /// Each subsequent bracket is penalized for similarity to existing ones
    pub fn generate_greedy_diverse(
        tournament: &TournamentInfo,
        num_brackets: usize,
        diversity_weight: f64,
        generations: u32,
    ) -> Self {
        let mut portfolio = BracketPortfolio::new();

        for i in 0..num_brackets {
            let bracket = if i == 0 {
                // First bracket: pure optimization
                optimize_bracket(tournament, generations)
            } else {
                // Subsequent brackets: optimize with diversity penalty
                optimize_with_diversity(tournament, &portfolio.brackets, diversity_weight, generations)
            };

            portfolio.brackets.push(bracket);
            portfolio.constraints.push(Vec::new()); // No explicit constraints
        }

        portfolio
    }

    /// Generate portfolio using Genetic Algorithm and Monte Carlo simulations
    /// Each bracket maximizes the overall portfolio fitness (max score of set)
    pub fn generate_genetic_portfolio(
        tournament: &TournamentInfo,
        num_brackets: usize,
        num_simulations: i32,
        generations: u32,
        mode: GeneticMode,
    ) -> Self {
        let mut portfolio = BracketPortfolio::new();

        println!("Generating simulation pool of size {}...", num_simulations);
        let simulation_pool = Batch::new(tournament, num_simulations);

        let config = GeneticConfig::new(generations);

        match mode {
            GeneticMode::Sequential => {
                println!("Running Genetic Algorithm in Sequential mode...");
                for i in 0..num_brackets {
                    println!("Optimizing bracket {} of {}...", i + 1, num_brackets);

                    let bracket = genetic::optimize_portfolio_bracket(
                        tournament,
                        &portfolio.brackets,
                        &simulation_pool,
                        &config
                    );

                    portfolio.brackets.push(bracket);
                    portfolio.constraints.push(Vec::new());
                }
            },
            GeneticMode::Simultaneous => {
                println!("Running Genetic Algorithm in Simultaneous mode (Whole Portfolio)...");
                let brackets = genetic::optimize_whole_portfolio(
                    tournament,
                    num_brackets,
                    &simulation_pool,
                    &config
                );
                portfolio.brackets = brackets;
                portfolio.constraints = vec![Vec::new(); num_brackets];
            }
        }

        portfolio
    }

    /// Calculate statistics about the portfolio
    pub fn stats(&self) -> PortfolioStats {
        if self.brackets.is_empty() {
            return PortfolioStats::default();
        }

        // Calculate average EV
        let avg_ev = self.brackets.iter()
            .map(|b| b.expected_value)
            .sum::<f64>() / self.brackets.len() as f64;

        // Calculate pairwise similarities
        let mut total_similarity = 0.0;
        let mut min_similarity = 1.0;
        let mut comparisons = 0;

        for i in 0..self.brackets.len() {
            for j in (i + 1)..self.brackets.len() {
                let dist = self.brackets[i].weighted_distance(&self.brackets[j]);
                total_similarity += dist.similarity;
                if dist.similarity < min_similarity {
                    min_similarity = dist.similarity;
                }
                comparisons += 1;
            }
        }

        let avg_similarity = if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            1.0
        };

        // Count unique champions
        let mut champions: Vec<String> = self.brackets.iter()
            .map(|b| b.winner.name.clone())
            .collect();
        champions.sort();
        champions.dedup();

        PortfolioStats {
            num_brackets: self.brackets.len(),
            avg_expected_value: avg_ev,
            avg_similarity,
            min_similarity,
            unique_champions: champions.len(),
            champion_names: champions,
        }
    }

    /// Pretty print portfolio summary
    pub fn print_summary(&self) {
        let stats = self.stats();

        println!("\n=== Bracket Portfolio Summary ===");
        println!("Number of brackets: {}", stats.num_brackets);
        println!("Average EV: {:.2}", stats.avg_expected_value);
        println!("Average similarity: {:.1}%", stats.avg_similarity * 100.0);
        println!("Minimum similarity: {:.1}%", stats.min_similarity * 100.0);
        println!("Unique champions: {}", stats.unique_champions);
        println!("Champions: {:?}", stats.champion_names);
        println!();

        for (i, bracket) in self.brackets.iter().enumerate() {
            println!("Bracket {}: Champion = {} (seed {}), EV = {:.2}",
                     i + 1, bracket.winner.name, bracket.winner.seed, bracket.expected_value);
        }
        println!();
    }

    /// Print detailed comparison between brackets
    pub fn print_pairwise_distances(&self) {
        println!("\n=== Pairwise Bracket Distances ===");
        for i in 0..self.brackets.len() {
            for j in (i + 1)..self.brackets.len() {
                let dist = self.brackets[i].weighted_distance(&self.brackets[j]);
                println!("Brackets {} vs {}: Similarity = {:.1}%, Champion match = {}",
                         i + 1, j + 1, dist.similarity * 100.0, dist.champion_match);
            }
        }
        println!();
    }

    /// Analyze how the portfolio performs in different scenarios
    pub fn analyze_scenarios(&self, simulation_pool: &Batch) {
        if self.brackets.is_empty() || simulation_pool.brackets.is_empty() {
            return;
        }

        println!("\n=== Scenario Analysis ===");

        let mut bracket_wins = HashMap::new();
        for i in 0..self.brackets.len() {
            bracket_wins.insert(i, 0);
        }

        let mut scenario_counts: HashMap<String, usize> = HashMap::new();
        let mut scenario_best_brackets: HashMap<String, Vec<usize>> = HashMap::new();

        // Categorize each simulation
        for sim in &simulation_pool.brackets {
            let winner_seed = sim.winner.seed;
            let scenario_type = if winner_seed <= 1 {
                "Favorities (1 Seeds)".to_string()
            } else if winner_seed <= 4 {
                "Contenders (2-4 Seeds)".to_string()
            } else {
                "Chaos (5+ Seeds)".to_string()
            };

            *scenario_counts.entry(scenario_type.clone()).or_insert(0) += 1;

            // Find which bracket in portfolio performs best for this specific simulation
            let mut best_score = -1.0;
            let mut best_bracket_indices = Vec::new();

            for (idx, bracket) in self.brackets.iter().enumerate() {
                let score = bracket.score(sim);
                if score > best_score {
                    best_score = score;
                    best_bracket_indices = vec![idx];
                } else if (score - best_score).abs() < 0.001 {
                    best_bracket_indices.push(idx);
                }
            }

            // Update win counts
            // If multiple brackets tie for best, fractional win? Or full win for each?
            // Let's give full win for "best performing in scenario"
            for &idx in &best_bracket_indices {
                *bracket_wins.get_mut(&idx).unwrap() += 1;

                scenario_best_brackets.entry(scenario_type.clone())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }

        // Print breakdown by bracket
        println!("\nBest Performing Bracket by Simulation Count:");
        for i in 0..self.brackets.len() {
            let wins = bracket_wins.get(&i).unwrap_or(&0);
            let pct = *wins as f64 / simulation_pool.brackets.len() as f64 * 100.0;
            println!("Bracket {}: Best in {} simulations ({:.1}%)", i + 1, wins, pct);
        }

        // Print breakdown by scenario
        println!("\nPerformance by Scenario Type:");
        let mut sorted_scenarios: Vec<_> = scenario_counts.keys().cloned().collect();
        sorted_scenarios.sort();

        for scenario in sorted_scenarios {
            let count = scenario_counts.get(&scenario).unwrap();
            let pct = *count as f64 / simulation_pool.brackets.len() as f64 * 100.0;

            println!("\nScenario: {} ({} sims, {:.1}%)", scenario, count, pct);

            // Calculate which bracket is best most often in this scenario
            let best_indices = scenario_best_brackets.get(&scenario).unwrap();
            let mut bracket_scenario_wins = HashMap::new();
            for &idx in best_indices {
                *bracket_scenario_wins.entry(idx).or_insert(0) += 1;
            }

            let mut sorted_wins: Vec<_> = bracket_scenario_wins.iter().collect();
            sorted_wins.sort_by(|a, b| b.1.cmp(a.1));

            for (idx, wins) in sorted_wins.iter().take(3) {
                let win_pct = **wins as f64 / *count as f64 * 100.0;
                println!("  Bracket {}: Best in {:.1}% of these scenarios", *idx + 1, win_pct);
            }
        }
        println!();
    }
}

#[derive(Debug, Clone, Default)]
pub struct PortfolioStats {
    pub num_brackets: usize,
    pub avg_expected_value: f64,
    pub avg_similarity: f64,
    pub min_similarity: f64,
    pub unique_champions: usize,
    pub champion_names: Vec<String>,
}

/// Optimize a single bracket using genetic algorithm
fn optimize_bracket(tournament: &TournamentInfo, generations: u32) -> Bracket {
    let mut bracket = Bracket::new(tournament);
    let mutation_rate = 1.0 / 63.0 * 3.0;

    for _ in 0..generations {
        let child = bracket.mutate(tournament, mutation_rate);
        if child.expected_value > bracket.expected_value {
            bracket = child;
        }
    }

    bracket
}

/// Optimize bracket with diversity penalty from existing brackets
fn optimize_with_diversity(
    tournament: &TournamentInfo,
    existing: &[Bracket],
    diversity_weight: f64,
    generations: u32,
) -> Bracket {
    let mut best_bracket = Bracket::new(tournament);
    let mut best_score = fitness_with_diversity(&best_bracket, existing, diversity_weight);
    let mutation_rate = 1.0 / 63.0 * 3.0;

    for _ in 0..generations {
        let child = best_bracket.mutate(tournament, mutation_rate);
        let child_score = fitness_with_diversity(&child, existing, diversity_weight);

        if child_score > best_score {
            best_bracket = child;
            best_score = child_score;
        }
    }

    best_bracket
}

/// Calculate fitness combining EV and diversity
fn fitness_with_diversity(bracket: &Bracket, existing: &[Bracket], diversity_weight: f64) -> f64 {
    let ev = bracket.expected_value;

    if existing.is_empty() {
        return ev;
    }

    // Calculate minimum distance to any existing bracket
    let min_distance = existing.iter()
        .map(|b| bracket.weighted_distance(b).total_distance)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // Higher distance = more diverse = better
    ev + diversity_weight * min_distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advancement_round_wins() {
        assert_eq!(AdvancementRound::Round2.wins_required(), 1);
        assert_eq!(AdvancementRound::FinalFour.wins_required(), 4);
        assert_eq!(AdvancementRound::Winner.wins_required(), 6);
    }
}
