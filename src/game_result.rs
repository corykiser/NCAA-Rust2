// This module defines the data structures for storing game results fetched from APIs
// These results are used to calculate ELO ratings

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Represents the result of a single basketball game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameResult {
    pub game_id: String,
    pub date: NaiveDate,
    pub home_team_id: String,
    pub home_team_name: String,
    pub away_team_id: String,
    pub away_team_name: String,
    pub home_score: u32,
    pub away_score: u32,
    pub is_neutral_site: bool,
    pub is_conference_game: bool,
    pub is_completed: bool,
}

impl GameResult {
    pub fn new(
        game_id: String,
        date: NaiveDate,
        home_team_id: String,
        home_team_name: String,
        away_team_id: String,
        away_team_name: String,
        home_score: u32,
        away_score: u32,
    ) -> Self {
        GameResult {
            game_id,
            date,
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name,
            home_score,
            away_score,
            is_neutral_site: false,
            is_conference_game: false,
            is_completed: true,
        }
    }

    /// Returns the winning team's ID
    pub fn winner_id(&self) -> &str {
        if self.home_score > self.away_score {
            &self.home_team_id
        } else {
            &self.away_team_id
        }
    }

    /// Returns the losing team's ID
    pub fn loser_id(&self) -> &str {
        if self.home_score > self.away_score {
            &self.away_team_id
        } else {
            &self.home_team_id
        }
    }

    /// Returns the winning team's name
    pub fn winner_name(&self) -> &str {
        if self.home_score > self.away_score {
            &self.home_team_name
        } else {
            &self.away_team_name
        }
    }

    /// Returns the margin of victory (positive value)
    pub fn margin(&self) -> u32 {
        if self.home_score > self.away_score {
            self.home_score - self.away_score
        } else {
            self.away_score - self.home_score
        }
    }

    /// Returns true if the home team won
    pub fn home_won(&self) -> bool {
        self.home_score > self.away_score
    }
}

/// Basic team information from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamInfo {
    pub id: String,
    pub name: String,
    pub abbreviation: String,
    pub conference: Option<String>,
    pub logo_url: Option<String>,
}

impl TeamInfo {
    pub fn new(id: String, name: String, abbreviation: String) -> Self {
        TeamInfo {
            id,
            name,
            abbreviation,
            conference: None,
            logo_url: None,
        }
    }
}

/// Tournament bracket team with seed and region info (for March Madness)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BracketTeam {
    pub team_id: String,
    pub team_name: String,
    pub seed: i32,
    pub region: String,
}

impl BracketTeam {
    pub fn new(team_id: String, team_name: String, seed: i32, region: String) -> Self {
        BracketTeam {
            team_id,
            team_name,
            seed,
            region,
        }
    }
}

/// Cache metadata for stored game results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameCache {
    pub season: String,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub games: Vec<GameResult>,
}

impl GameCache {
    pub fn new(season: String, games: Vec<GameResult>) -> Self {
        GameCache {
            season,
            last_updated: chrono::Utc::now(),
            games,
        }
    }

    /// Returns true if the cache is older than the specified hours
    pub fn is_stale(&self, hours: i64) -> bool {
        let age = chrono::Utc::now() - self.last_updated;
        age.num_hours() >= hours
    }
}
