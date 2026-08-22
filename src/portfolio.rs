// Portfolio generation for diverse bracket strategies
// Supports constrained bracket building and champion-stratified portfolios

use crate::anneal::{self, AnnealingConfig};
use crate::bracket::{Bracket, ScoringConfig};
use crate::exact::{self, TeamLock};
use crate::ga::LockSet;
use crate::ingest::{RcTeam, TournamentInfo};
use serde::{Deserialize, Serialize};

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
/// Builds a bracket that satisfies a set of team-advancement constraints.
///
/// This used to write winners directly into `bracket.games[i].winner` without
/// touching the games downstream, the 63-bit representation, or the win
/// probabilities. The result was a bracket whose Elite 8 and Final Four
/// disagreed — a team could appear in the Final Four having won no Elite 8
/// game — and whose `binary` still described the pre-constraint bracket, so the
/// constraint vanished the first time anything mutated it.
///
/// Constraints are now applied to the bit vector and the bracket is rebuilt
/// from it, which makes propagation automatic and the result legal by
/// construction. The build is verified against the constraints before it is
/// returned.
pub struct ConstrainedBracketBuilder<'a> {
    tournament: &'a TournamentInfo,
    scoring_config: &'a ScoringConfig,
    constraints: Vec<BracketConstraint>,
}

impl<'a> ConstrainedBracketBuilder<'a> {
    pub fn new(tournament: &'a TournamentInfo, scoring_config: &'a ScoringConfig) -> Self {
        ConstrainedBracketBuilder {
            tournament,
            scoring_config,
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

    /// Resolve the constraints to team locks, rejecting names that do not
    /// identify exactly one team and locks that cannot hold together.
    pub fn locks(&self) -> Result<LockSet, String> {
        let mut locks = Vec::with_capacity(self.constraints.len());
        for constraint in &self.constraints {
            let team = self
                .tournament
                .find_team(&constraint.team_name)
                .map_err(|e| e.to_string())?;
            locks.push(TeamLock {
                team_index: team.team_index,
                wins_required: constraint.must_reach.wins_required(),
            });
        }

        // Surfaces "both of these teams must win the same game" before the
        // caller spends a run discovering only one of them made it.
        exact::forced_winners(self.tournament, &locks).map_err(|e| e.to_string())?;

        Ok(LockSet::new(locks))
    }

    /// Build a bracket satisfying the constraints, with the unconstrained games
    /// sampled from the rating model.
    pub fn build(&self) -> Result<Bracket, String> {
        let locks = self.locks()?;
        let base = Bracket::new(self.tournament, Some(self.scoring_config));
        let bracket = locks.repair(base, self.tournament, self.scoring_config);
        self.verify(&bracket)?;
        Ok(bracket)
    }

    /// Build the expected-value-maximizing bracket satisfying the constraints.
    ///
    /// Exact, not a search: see `crate::exact`.
    pub fn build_optimal(&self) -> Result<Bracket, String> {
        let locks = self.locks()?;
        let solution = exact::solve(self.tournament, self.scoring_config, &locks.locks)
            .map_err(|e| e.to_string())?;
        self.verify(&solution.bracket)?;
        Ok(solution.bracket)
    }

    /// Confirm every constraint actually holds. A failure here is a bug in the
    /// constraint machinery, not bad user input, so it reports loudly rather
    /// than returning a bracket that quietly ignores what was asked for.
    fn verify(&self, bracket: &Bracket) -> Result<(), String> {
        for constraint in &self.constraints {
            let team = self
                .tournament
                .find_team(&constraint.team_name)
                .map_err(|e| e.to_string())?;
            let wins = bracket.wins_for(team.team_index);
            let required = constraint.must_reach.wins_required();
            if wins < required {
                return Err(format!(
                    "constraint not satisfied: {} wins {} game(s) but must win {} to reach {:?}",
                    team.name, wins, required, constraint.must_reach
                ));
            }
        }
        Ok(())
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
        scoring_config: &ScoringConfig,
    ) -> Self {
        let mut portfolio = BracketPortfolio::new();

        // Rank teams by rating (proxy for championship probability)
        // RcTeam dereferences to Team, so we can access .rating directly
        let mut ranked_teams: Vec<&RcTeam> = tournament.teams.iter().collect();
        ranked_teams.sort_by(|a, b| b.rating.partial_cmp(&a.rating).unwrap());

        // Generate one bracket per top team
        for team in ranked_teams.iter().take(num_brackets) {
            let constraint = BracketConstraint::champion(&team.name);

            let builder = ConstrainedBracketBuilder::new(tournament, scoring_config)
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
        scoring_config: &ScoringConfig,
    ) -> Self {
        let mut portfolio = BracketPortfolio::new();

        for i in 0..num_brackets {
            let bracket = if i == 0 {
                // First bracket: pure optimization
                optimize_bracket(tournament, generations, scoring_config)
            } else {
                // Subsequent brackets: optimize with diversity penalty
                optimize_with_diversity(tournament, &portfolio.brackets, diversity_weight, generations, scoring_config)
            };

            portfolio.brackets.push(bracket);
            portfolio.constraints.push(Vec::new()); // No explicit constraints
        }

        portfolio
    }

    /// Generate portfolio using Simulated Annealing optimization
    /// Uses best-ball scoring against a Monte Carlo simulation pool
    pub fn generate_annealing_diverse(
        tournament: &TournamentInfo,
        num_brackets: usize,
        pool_size: usize,
        steps: usize,
        scoring_config: &ScoringConfig,
    ) -> Self {
        let config = AnnealingConfig {
            steps,
            pool_size,
            ..Default::default()
        };

        anneal::optimize_portfolio(tournament, num_brackets, config, scoring_config)
    }

    /// Calculate statistics about the portfolio
    pub fn stats(&self, scoring_config: Option<&ScoringConfig>) -> PortfolioStats {
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
                // Use the provided scoring config for weighted distance stats
                // If None, it will use the default config inside weighted_distance, which is a fallback
                // but ideally the caller should pass the correct config.
                let dist = self.brackets[i].weighted_distance(&self.brackets[j], scoring_config);
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
        // Warning: This prints stats using default scoring if called without config
        // But the method signature doesn't take config for now to avoid breaking other calls if any.
        // Wait, I can update the signature since I'm editing the file.
        // But main.rs calls it. I should update main.rs too or just make stats take Option.
        // Let's make stats take Option<&ScoringConfig> and print_summary rely on that.
        // For print_summary, I probably can't easily change the signature without updating callers.
        // Let's see: main.rs calls it. I can update main.rs.

        // Actually, print_summary uses self.stats().
        // I'll update print_summary to take the config.
        eprintln!("Warning: print_summary called without config, using defaults for stats.");
        self.print_summary_with_config(None);
    }

    pub fn print_summary_with_config(&self, scoring_config: Option<&ScoringConfig>) {
        let stats = self.stats(scoring_config);

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
    pub fn print_pairwise_distances(&self, scoring_config: &ScoringConfig) {
        println!("\n=== Pairwise Bracket Distances ===");
        for i in 0..self.brackets.len() {
            for j in (i + 1)..self.brackets.len() {
                // Use actual scoring config for distance
                let dist = self.brackets[i].weighted_distance(&self.brackets[j], Some(scoring_config));
                println!("Brackets {} vs {}: Similarity = {:.1}%, Champion match = {}",
                         i + 1, j + 1, dist.similarity * 100.0, dist.champion_match);
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
fn optimize_bracket(tournament: &TournamentInfo, generations: u32, scoring_config: &ScoringConfig) -> Bracket {
    let mut bracket = Bracket::new(tournament, Some(scoring_config));
    let mutation_rate = 1.0 / 63.0 * 3.0;

    for _ in 0..generations {
        let child = bracket.mutate(tournament, mutation_rate, Some(scoring_config));
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
    scoring_config: &ScoringConfig,
) -> Bracket {
    let mut best_bracket = Bracket::new(tournament, Some(scoring_config));
    let mut best_score = fitness_with_diversity(&best_bracket, existing, diversity_weight, scoring_config);
    let mutation_rate = 1.0 / 63.0 * 3.0;

    for _ in 0..generations {
        let child = best_bracket.mutate(tournament, mutation_rate, Some(scoring_config));
        let child_score = fitness_with_diversity(&child, existing, diversity_weight, scoring_config);

        if child_score > best_score {
            best_bracket = child;
            best_score = child_score;
        }
    }

    best_bracket
}

/// Calculate fitness combining EV and diversity
fn fitness_with_diversity(bracket: &Bracket, existing: &[Bracket], diversity_weight: f64, scoring_config: &ScoringConfig) -> f64 {
    let ev = bracket.expected_value;

    if existing.is_empty() {
        return ev;
    }

    // Calculate minimum distance to any existing bracket
    let min_distance = existing.iter()
        .map(|b| bracket.weighted_distance(b, Some(scoring_config)).total_distance)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // Higher distance = more diverse = better
    ev + diversity_weight * min_distance
}

#[cfg(test)]
mod constrained_tests {
    use super::*;
    use crate::bracket::tests::assert_legal;
    use crate::ingest::tests::tournament;

    #[test]
    fn a_constrained_bracket_is_internally_consistent() {
        // This is the regression for the bug where applying a constraint
        // overwrote a game's winner without updating the games it feeds. The
        // Elite 8 said one team and the Final Four showed another that had won
        // nothing — an impossible bracket, printed as if it were real.
        let t = tournament();
        let scoring = ScoringConfig::default();

        for seed in [1, 4, 8, 12, 16] {
            for round in [
                AdvancementRound::Round2,
                AdvancementRound::Sweet16,
                AdvancementRound::Elite8,
                AdvancementRound::FinalFour,
                AdvancementRound::Championship,
                AdvancementRound::Winner,
            ] {
                let team = t.teams.iter().find(|x| x.seed == seed).unwrap().clone();
                let bracket = ConstrainedBracketBuilder::new(&t, &scoring)
                    .with_constraint(BracketConstraint::new(&team.name, round))
                    .build()
                    .unwrap_or_else(|e| panic!("{} to {:?}: {}", team.name, round, e));

                assert_legal(&bracket);
                assert!(
                    bracket.wins_for(team.team_index) >= round.wins_required(),
                    "{} reached only {} wins, needed {} for {:?}",
                    team.name,
                    bracket.wins_for(team.team_index),
                    round.wins_required(),
                    round
                );
            }
        }
    }

    #[test]
    fn the_binary_representation_agrees_with_the_games() {
        // The constraint used to be written into `games` but not `binary`, so
        // rebuilding from the bits threw it away.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let team = t.teams.iter().find(|x| x.seed == 10).unwrap().clone();

        let bracket = ConstrainedBracketBuilder::new(&t, &scoring)
            .with_final_four(&team.name)
            .build()
            .unwrap();

        let rebuilt = Bracket::new_from_binary(&t, &bracket.binary, Some(&scoring));
        assert_eq!(rebuilt.winner_indices(), bracket.winner_indices());
        assert!(rebuilt.wins_for(team.team_index) >= 4);
    }

    #[test]
    fn the_optimal_constrained_bracket_beats_a_sampled_one() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let team = t.teams.iter().find(|x| x.seed == 9).unwrap().clone();
        let builder = ConstrainedBracketBuilder::new(&t, &scoring).with_champion(&team.name);

        let optimal = builder.build_optimal().unwrap();
        assert_legal(&optimal);
        assert_eq!(optimal.winner.team_index, team.team_index);

        for _ in 0..25 {
            let sampled = builder.build().unwrap();
            assert!(sampled.expected_value <= optimal.expected_value + 1e-9);
        }
    }

    #[test]
    fn an_unknown_or_ambiguous_team_is_an_error() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        assert!(ConstrainedBracketBuilder::new(&t, &scoring)
            .with_champion("Nowhere Polytechnic")
            .build()
            .is_err());
    }

    #[test]
    fn conflicting_constraints_are_rejected() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let a = t.teams.iter().find(|x| x.seed == 1).unwrap().clone();
        let b = t.teams.iter().find(|x| x.seed == 16).unwrap().clone();
        // Same round-1 game; they cannot both win it.
        let err = ConstrainedBracketBuilder::new(&t, &scoring)
            .with_constraint(BracketConstraint::new(&a.name, AdvancementRound::Round2))
            .with_constraint(BracketConstraint::new(&b.name, AdvancementRound::Round2))
            .build()
            .unwrap_err();
        assert!(err.contains("cannot both win"), "{}", err);
    }
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
