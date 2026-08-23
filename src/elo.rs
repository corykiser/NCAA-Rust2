// This module implements the ELO rating system for NCAA basketball teams
// ELO ratings are calculated from historical game results and used to predict future matchups

use crate::game_result::GameResult;
use crate::names;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default starting ELO rating for all teams
pub const DEFAULT_RATING: f64 = 1500.0;

/// Home court advantage in ELO points
pub const HOME_ADVANTAGE: f64 = 100.0;

/// K-factor for early season (first 10 games)
pub const K_FACTOR_EARLY: f64 = 32.0;

/// K-factor for late season (after 10 games)
pub const K_FACTOR_LATE: f64 = 20.0;

/// Elo points per point of 538 rating. The tournament's win-probability model
/// (`ingest::ProbabilityCache`) exponentiates `rating_diff * 30.464 / 400`,
/// which is the standard Elo curve with 538's rating scale folded in.
pub const ELO_PER_538_POINT: f64 = 30.464;
/// The two scales are anchored at the Elo starting rating: a 1500 is a 75.
const SCALE_ANCHOR_ELO: f64 = 1500.0;
/// The 538 rating of an average team. Public because `ratings.rs` anchors its
/// own scale to the same point, which is what lets the two rating sources be
/// raced against each other on one field.
pub const SCALE_ANCHOR_538: f64 = 75.0;

/// Represents a team's ELO rating and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloRating {
    pub team_id: String,
    pub team_name: String,
    pub rating: f64,
    pub games_played: u32,
    pub wins: u32,
    pub losses: u32,
}

impl EloRating {
    pub fn new(team_id: String, team_name: String) -> Self {
        EloRating {
            team_id,
            team_name,
            rating: DEFAULT_RATING,
            games_played: 0,
            wins: 0,
            losses: 0,
        }
    }

    /// Returns the win percentage as a value between 0 and 1
    pub fn win_pct(&self) -> f64 {
        if self.games_played == 0 {
            0.5
        } else {
            self.wins as f64 / self.games_played as f64
        }
    }

    /// Returns the appropriate K-factor based on games played
    pub fn k_factor(&self) -> f64 {
        if self.games_played < 10 {
            K_FACTOR_EARLY
        } else {
            K_FACTOR_LATE
        }
    }
}

/// Manages ELO ratings for all teams in a season
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloSystem {
    pub ratings: HashMap<String, EloRating>,
    pub games_processed: u32,
    pub season: String,
}

impl EloSystem {
    pub fn new(season: String) -> Self {
        EloSystem {
            ratings: HashMap::new(),
            games_processed: 0,
            season,
        }
    }

    /// Initialize a team with default rating if not already present
    pub fn ensure_team(&mut self, team_id: &str, team_name: &str) {
        if !self.ratings.contains_key(team_id) {
            self.ratings.insert(
                team_id.to_string(),
                EloRating::new(team_id.to_string(), team_name.to_string()),
            );
        }
    }

    /// Calculate expected score (probability of winning) for team A vs team B
    /// Uses the standard ELO formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    pub fn expected_score(rating_a: f64, rating_b: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
    }

    /// Calculate margin of victory multiplier
    /// This gives more weight to blowout wins and less to close games
    /// Formula based on FiveThirtyEight's approach
    pub fn mov_multiplier(winner_elo: f64, loser_elo: f64, margin: u32) -> f64 {
        let elo_diff = winner_elo - loser_elo;
        let margin_f = margin as f64;

        // Log-based multiplier that diminishes for larger margins
        let base_mult = (margin_f + 3.0).ln() / (3.0_f64.ln());

        // Reduce multiplier when favorite wins big (expected), increase when underdog wins
        if elo_diff > 0.0 {
            // Favorite won - reduce bonus
            base_mult * (2.2 / (elo_diff * 0.001 + 2.2))
        } else {
            // Underdog won - apply bonus
            base_mult
        }
    }

    /// Process a single game and update both teams' ratings
    pub fn process_game(&mut self, game: &GameResult) {
        // Ensure both teams exist in the system
        self.ensure_team(&game.home_team_id, &game.home_team_name);
        self.ensure_team(&game.away_team_id, &game.away_team_name);

        // Get current ratings
        let home_rating = self.ratings.get(&game.home_team_id).unwrap().rating;
        let away_rating = self.ratings.get(&game.away_team_id).unwrap().rating;

        // Apply home court advantage (unless neutral site)
        let effective_home_rating = if game.is_neutral_site {
            home_rating
        } else {
            home_rating + HOME_ADVANTAGE
        };

        // Calculate expected scores
        let home_expected = Self::expected_score(effective_home_rating, away_rating);
        let away_expected = 1.0 - home_expected;

        // Actual scores (1 for win, 0 for loss)
        let (home_actual, away_actual) = if game.home_won() {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };

        // Get K-factors for each team
        let home_k = self.ratings.get(&game.home_team_id).unwrap().k_factor();
        let away_k = self.ratings.get(&game.away_team_id).unwrap().k_factor();
        let k_factor = (home_k + away_k) / 2.0; // Average K-factor

        // Calculate margin of victory multiplier
        let (winner_rating, loser_rating) = if game.home_won() {
            (home_rating, away_rating)
        } else {
            (away_rating, home_rating)
        };
        let mov_mult = Self::mov_multiplier(winner_rating, loser_rating, game.margin());

        // Calculate rating changes with MOV multiplier
        let home_change = k_factor * mov_mult * (home_actual - home_expected);
        let away_change = k_factor * mov_mult * (away_actual - away_expected);

        // Update ratings
        {
            let home_team = self.ratings.get_mut(&game.home_team_id).unwrap();
            home_team.rating += home_change;
            home_team.games_played += 1;
            if game.home_won() {
                home_team.wins += 1;
            } else {
                home_team.losses += 1;
            }
        }

        {
            let away_team = self.ratings.get_mut(&game.away_team_id).unwrap();
            away_team.rating += away_change;
            away_team.games_played += 1;
            if !game.home_won() {
                away_team.wins += 1;
            } else {
                away_team.losses += 1;
            }
        }

        self.games_processed += 1;
    }

    /// Process multiple games in chronological order
    pub fn process_games(&mut self, games: &mut [GameResult]) {
        // Sort games by date
        games.sort_by(|a, b| a.date.cmp(&b.date));

        for game in games.iter() {
            if game.is_completed {
                self.process_game(game);
            }
        }
    }

    /// Get a team's current rating (returns default if team not found)
    pub fn get_rating(&self, team_id: &str) -> f64 {
        self.ratings
            .get(team_id)
            .map(|r| r.rating)
            .unwrap_or(DEFAULT_RATING)
    }

    /// Get the top N teams by rating
    pub fn top_teams(&self, n: usize) -> Vec<&EloRating> {
        let mut teams: Vec<&EloRating> = self.ratings.values().collect();
        teams.sort_by(|a, b| b.rating.partial_cmp(&a.rating).unwrap());
        teams.into_iter().take(n).collect()
    }

    /// Print the top N teams with their ratings
    pub fn print_top_teams(&self, n: usize) {
        println!("\nTop {} Teams by ELO Rating:", n);
        println!("{:>4} {:>6} {:<25} {:>5} {:>5}", "Rank", "Rating", "Team", "W", "L");
        println!("{}", "-".repeat(50));

        for (i, team) in self.top_teams(n).iter().enumerate() {
            println!(
                "{:>4} {:>6.1} {:<25} {:>5} {:>5}",
                i + 1,
                team.rating,
                &team.team_name[..team.team_name.len().min(25)],
                team.wins,
                team.losses
            );
        }
    }

    /// Convert ELO rating to the scale used by 538 (roughly 0-100)
    /// This helps with compatibility with existing bracket simulation
    /// Convert an Elo rating to the 538 rating scale the tournament model uses.
    ///
    /// The two are the same logistic curve at different widths: `ProbabilityCache`
    /// scales a 538 rating difference by `30.464/400` before exponentiating, so
    /// one point of 538 rating is `ELO_PER_538_POINT` points of Elo. The
    /// conversion is therefore linear, and every matchup keeps the win
    /// probability the Elo ratings implied.
    ///
    /// This used to map 1200-1800 Elo onto 60-100 and clamp. Ratings from a
    /// full season run well past 1800 — the 2025-26 field topped out above
    /// 2000 — so nearly every tournament team clamped to exactly 100.0. Every
    /// game became a coin flip, and with seed bonuses in the scoring rules the
    /// "optimal" bracket was whichever one had the highest seeds in it: a 16
    /// over a 1, and no 1-seed in the Final Four.
    pub fn to_538_scale(&self, team_id: &str) -> f32 {
        let elo = self.get_rating(team_id);
        (SCALE_ANCHOR_538 + (elo - SCALE_ANCHOR_ELO) / ELO_PER_538_POINT) as f32
    }

    /// Find a rated team by name.
    ///
    /// This used to be a `.contains()` scan over `self.ratings.values()`. Two
    /// problems: "Texas" matched "Texas A&M" and "Texas Tech" as readily as
    /// "Texas", and `HashMap` iteration order is unspecified, so which one it
    /// returned varied between runs of the same binary. Resolution now goes
    /// through `names::resolve`, which prefers exact matches and reports an
    /// error instead of picking arbitrarily among equally good candidates.
    pub fn find_team_by_name(&self, name: &str) -> Result<&EloRating, names::NameError> {
        let mut rated: Vec<&EloRating> = self.ratings.values().collect();
        // Deterministic order in, deterministic error messages out.
        rated.sort_by(|a, b| a.team_id.cmp(&b.team_id));

        let candidates: Vec<String> = rated.iter().map(|r| r.team_name.clone()).collect();
        names::resolve(name, &candidates).map(|i| rated[i])
    }
}

/// Calculate win probability between two teams given their ELO ratings
/// This matches the formula used in bracket.rs for consistency
pub fn win_probability(rating_a: f32, rating_b: f32) -> f64 {
    let rating_diff = rating_a as f64 - rating_b as f64;
    1.0 / (1.0 + 10.0_f64.powf(-rating_diff * 30.464 / 400.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    /// The 538-scale conversion has to preserve win probabilities: converting
    /// two Elo ratings and running them through the tournament's probability
    /// model must give the same answer as the Elo curve itself.
    #[test]
    fn the_538_conversion_preserves_win_probability() {
        let mut elo = EloSystem::new("test".to_string());
        elo.ensure_team("strong", "Strong");
        elo.ensure_team("weak", "Weak");
        elo.ratings.get_mut("strong").unwrap().rating = 1996.8;
        elo.ratings.get_mut("weak").unwrap().rating = 1550.0;

        let elo_diff = 1996.8 - 1550.0;
        let from_elo = 1.0 / (1.0 + 10f64.powf(-elo_diff / 400.0));

        let diff_538 = (elo.to_538_scale("strong") - elo.to_538_scale("weak")) as f64;
        let from_538 = 1.0 / (1.0 + 10f64.powf(-diff_538 * ELO_PER_538_POINT / 400.0));

        assert!(
            (from_elo - from_538).abs() < 1e-4,
            "{} vs {}",
            from_elo,
            from_538
        );
    }

    /// The whole 2025-26 tournament field sat above the old 1200-1800 window,
    /// so every team converted to exactly 100.0 and every game to a coin flip.
    #[test]
    fn strong_teams_do_not_all_convert_to_the_same_rating() {
        let mut elo = EloSystem::new("test".to_string());
        for (id, rating) in [("a", 2035.2), ("b", 1996.8), ("c", 1850.0)] {
            elo.ensure_team(id, id);
            elo.ratings.get_mut(id).unwrap().rating = rating;
        }
        let a = elo.to_538_scale("a");
        let b = elo.to_538_scale("b");
        let c = elo.to_538_scale("c");
        assert!(a > b && b > c, "{} {} {}", a, b, c);
    }

    #[test]
    fn test_expected_score() {
        // Equal ratings should give 0.5
        let score = EloSystem::expected_score(1500.0, 1500.0);
        assert!((score - 0.5).abs() < 0.001);

        // Higher rating should give > 0.5
        let score = EloSystem::expected_score(1600.0, 1500.0);
        assert!(score > 0.5);

        // Much higher rating should give close to 1.0
        let score = EloSystem::expected_score(1800.0, 1200.0);
        assert!(score > 0.9);
    }

    #[test]
    fn test_rating_update() {
        let mut system = EloSystem::new("2024-2025".to_string());

        let game = GameResult::new(
            "test1".to_string(),
            NaiveDate::from_ymd_opt(2024, 11, 1).unwrap(),
            "team_a".to_string(),
            "Team A".to_string(),
            "team_b".to_string(),
            "Team B".to_string(),
            80,
            70,
        );

        system.process_game(&game);

        // Winner should gain rating
        assert!(system.get_rating("team_a") > DEFAULT_RATING);
        // Loser should lose rating
        assert!(system.get_rating("team_b") < DEFAULT_RATING);
        // Changes should be symmetric
        let a_change = system.get_rating("team_a") - DEFAULT_RATING;
        let b_change = DEFAULT_RATING - system.get_rating("team_b");
        assert!((a_change - b_change).abs() < 1.0);
    }
}
