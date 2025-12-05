// This module handles fetching game data from ESPN and NCAA APIs
// to calculate ELO ratings for NCAA basketball teams

use crate::game_result::{GameCache, GameResult, TeamInfo};
use chrono::{Datelike, NaiveDate};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;

/// Data source for fetching game results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataSource {
    ESPN,
    NCAA,
}

/// ESPN API base URL
const ESPN_BASE_URL: &str = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball";

/// NCAA API base URL (henrygd)
const NCAA_BASE_URL: &str = "https://ncaa-api.henrygd.me";

/// Rate limit delay in milliseconds (for NCAA API: 5 req/sec max)
const RATE_LIMIT_DELAY_MS: u64 = 250;

/// API client for fetching NCAA basketball data
pub struct ApiClient {
    client: reqwest::blocking::Client,
    source: DataSource,
    cache_dir: String,
}

impl ApiClient {
    pub fn new(source: DataSource, cache_dir: &str) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("NCAA-Bracket-Optimizer/1.0")
            .build()
            .expect("Failed to create HTTP client");

        // Create cache directory if it doesn't exist
        fs::create_dir_all(cache_dir).ok();

        ApiClient {
            client,
            source,
            cache_dir: cache_dir.to_string(),
        }
    }

    /// Fetch all teams from ESPN API
    pub fn fetch_all_teams(&self) -> Result<Vec<TeamInfo>, String> {
        match self.source {
            DataSource::ESPN => self.fetch_espn_teams(),
            DataSource::NCAA => self.fetch_ncaa_teams(),
        }
    }

    /// Fetch teams from ESPN
    fn fetch_espn_teams(&self) -> Result<Vec<TeamInfo>, String> {
        let url = format!("{}/teams?limit=400", ESPN_BASE_URL);
        let response = self.client.get(&url).send().map_err(|e| e.to_string())?;
        let json: Value = response.json().map_err(|e| e.to_string())?;

        let mut teams = Vec::new();

        if let Some(sports) = json.get("sports").and_then(|s| s.as_array()) {
            for sport in sports {
                if let Some(leagues) = sport.get("leagues").and_then(|l| l.as_array()) {
                    for league in leagues {
                        if let Some(team_arr) = league.get("teams").and_then(|t| t.as_array()) {
                            for team_obj in team_arr {
                                if let Some(team) = team_obj.get("team") {
                                    let id = team.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                                    let name = team.get("displayName").and_then(|n| n.as_str()).unwrap_or("").to_string();
                                    let abbr = team.get("abbreviation").and_then(|a| a.as_str()).unwrap_or("").to_string();

                                    if !id.is_empty() && !name.is_empty() {
                                        let mut team_info = TeamInfo::new(id, name, abbr);

                                        // Extract conference if available
                                        if let Some(groups) = team.get("groups") {
                                            if let Some(conf) = groups.get("parent").and_then(|p| p.get("name")).and_then(|n| n.as_str()) {
                                                team_info.conference = Some(conf.to_string());
                                            }
                                        }

                                        teams.push(team_info);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(teams)
    }

    /// Fetch teams from NCAA API
    fn fetch_ncaa_teams(&self) -> Result<Vec<TeamInfo>, String> {
        // NCAA API doesn't have a direct teams endpoint
        // We'll build the team list from game data instead
        Err("NCAA API doesn't have a teams endpoint. Teams are discovered from game data.".to_string())
    }

    /// Fetch games for a specific date from ESPN
    pub fn fetch_games_for_date(&self, date: NaiveDate) -> Result<Vec<GameResult>, String> {
        match self.source {
            DataSource::ESPN => self.fetch_espn_games_for_date(date),
            DataSource::NCAA => self.fetch_ncaa_games_for_date(date),
        }
    }

    /// Fetch games from ESPN for a specific date
    fn fetch_espn_games_for_date(&self, date: NaiveDate) -> Result<Vec<GameResult>, String> {
        let date_str = date.format("%Y%m%d").to_string();
        let url = format!(
            "{}/scoreboard?dates={}&groups=50&limit=500",
            ESPN_BASE_URL, date_str
        );

        let response = self.client.get(&url).send().map_err(|e| e.to_string())?;
        let json: Value = response.json().map_err(|e| e.to_string())?;

        let mut games = Vec::new();

        if let Some(events) = json.get("events").and_then(|e| e.as_array()) {
            for event in events {
                if let Some(game) = self.parse_espn_event(event, date) {
                    games.push(game);
                }
            }
        }

        // Rate limiting
        thread::sleep(Duration::from_millis(RATE_LIMIT_DELAY_MS));

        Ok(games)
    }

    /// Parse a single ESPN event into a GameResult
    fn parse_espn_event(&self, event: &Value, date: NaiveDate) -> Option<GameResult> {
        let game_id = event.get("id")?.as_str()?.to_string();

        // Check if game is completed
        let status = event.get("status")?.get("type")?;
        let is_completed = status.get("completed")?.as_bool()?;

        if !is_completed {
            return None;
        }

        // Get competitions array (usually just one)
        let competitions = event.get("competitions")?.as_array()?;
        let competition = competitions.first()?;

        // Check for neutral site
        let is_neutral = competition
            .get("neutralSite")
            .and_then(|n| n.as_bool())
            .unwrap_or(false);

        // Check for conference game
        let is_conference = competition
            .get("conferenceCompetition")
            .and_then(|c| c.as_bool())
            .unwrap_or(false);

        // Get competitors
        let competitors = competition.get("competitors")?.as_array()?;
        if competitors.len() != 2 {
            return None;
        }

        let mut home_team_id = String::new();
        let mut home_team_name = String::new();
        let mut home_score: u32 = 0;
        let mut away_team_id = String::new();
        let mut away_team_name = String::new();
        let mut away_score: u32 = 0;

        for competitor in competitors {
            let team = competitor.get("team")?;
            let team_id = team.get("id")?.as_str()?.to_string();
            let team_name = team.get("displayName").or(team.get("name"))?.as_str()?.to_string();
            let score: u32 = competitor.get("score")?.as_str()?.parse().ok()?;
            let home_away = competitor.get("homeAway")?.as_str()?;

            if home_away == "home" {
                home_team_id = team_id;
                home_team_name = team_name;
                home_score = score;
            } else {
                away_team_id = team_id;
                away_team_name = team_name;
                away_score = score;
            }
        }

        // Skip games with 0-0 scores (incomplete data)
        if home_score == 0 && away_score == 0 {
            return None;
        }

        Some(GameResult {
            game_id,
            date,
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name,
            home_score,
            away_score,
            is_neutral_site: is_neutral,
            is_conference_game: is_conference,
            is_completed: true,
        })
    }

    /// Fetch games from NCAA API for a specific date
    fn fetch_ncaa_games_for_date(&self, date: NaiveDate) -> Result<Vec<GameResult>, String> {
        let year = date.year();
        let month = date.month();
        let day = date.day();

        // NCAA API uses week format for scoreboard
        // For now, we'll use the schedule endpoint
        let url = format!(
            "{}/scoreboard/basketball-men/d1/{}/{}",
            NCAA_BASE_URL, year, month
        );

        let response = self.client.get(&url).send().map_err(|e| e.to_string())?;
        let json: Value = response.json().map_err(|e| e.to_string())?;

        let mut games = Vec::new();

        if let Some(game_arr) = json.get("games").and_then(|g| g.as_array()) {
            for game_data in game_arr {
                if let Some(game_obj) = game_data.get("game") {
                    // Parse game date
                    if let Some(game_date_str) = game_obj.get("startDate").and_then(|d| d.as_str()) {
                        // Check if this game is on the requested date
                        if let Ok(game_date) = NaiveDate::parse_from_str(&game_date_str[..10], "%m-%d-%Y") {
                            if game_date.day() == day {
                                if let Some(game) = self.parse_ncaa_game(game_obj, game_date) {
                                    games.push(game);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Rate limiting
        thread::sleep(Duration::from_millis(RATE_LIMIT_DELAY_MS));

        Ok(games)
    }

    /// Parse a single NCAA API game into a GameResult
    fn parse_ncaa_game(&self, game: &Value, date: NaiveDate) -> Option<GameResult> {
        let game_id = game.get("gameID")?.as_str()?.to_string();

        // Check if game is final
        let state = game.get("gameState")?.as_str()?;
        if state != "final" {
            return None;
        }

        // Get home team info
        let home = game.get("home")?;
        let home_team_name = home.get("names")?.get("full")?.as_str()?.to_string();
        let home_team_id = home.get("names")?.get("seo")?.as_str()?.to_string();
        let home_score: u32 = home.get("score")?.as_str()?.parse().ok()?;

        // Get away team info
        let away = game.get("away")?;
        let away_team_name = away.get("names")?.get("full")?.as_str()?.to_string();
        let away_team_id = away.get("names")?.get("seo")?.as_str()?.to_string();
        let away_score: u32 = away.get("score")?.as_str()?.parse().ok()?;

        Some(GameResult {
            game_id,
            date,
            home_team_id,
            home_team_name,
            away_team_id,
            away_team_name,
            home_score,
            away_score,
            is_neutral_site: false, // NCAA API doesn't provide this
            is_conference_game: false, // Would need to check conference
            is_completed: true,
        })
    }

    /// Fetch all games for a date range
    pub fn fetch_games_for_range(
        &self,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<GameResult>, String> {
        let mut all_games = Vec::new();
        let mut current_date = start_date;
        let total_days = (end_date - start_date).num_days() + 1;
        let mut days_processed = 0;

        println!("Fetching games from {} to {}...", start_date, end_date);

        while current_date <= end_date {
            match self.fetch_games_for_date(current_date) {
                Ok(games) => {
                    let game_count = games.len();
                    all_games.extend(games);
                    if game_count > 0 {
                        print!(".");
                    }
                }
                Err(e) => {
                    eprintln!("\nWarning: Failed to fetch games for {}: {}", current_date, e);
                }
            }

            days_processed += 1;
            if days_processed % 30 == 0 {
                println!(" [{}/{}]", days_processed, total_days);
            }

            current_date = current_date.succ_opt().unwrap();
        }

        println!("\nFetched {} games total", all_games.len());
        Ok(all_games)
    }

    /// Fetch all games for a college basketball season
    /// Season format: "2024-2025" means the season starting in Nov 2024
    pub fn fetch_season(&self, season: &str) -> Result<Vec<GameResult>, String> {
        // Parse season string
        let parts: Vec<&str> = season.split('-').collect();
        if parts.len() != 2 {
            return Err("Season must be in format YYYY-YYYY (e.g., 2024-2025)".to_string());
        }

        let start_year: i32 = parts[0].parse().map_err(|_| "Invalid start year")?;
        let end_year: i32 = parts[1].parse().map_err(|_| "Invalid end year")?;

        // College basketball season runs from early November to early April
        let start_date = NaiveDate::from_ymd_opt(start_year, 11, 4).unwrap();
        let end_date = NaiveDate::from_ymd_opt(end_year, 4, 10).unwrap();

        // Check cache first
        let cache_path = format!("{}/games_{}.json", self.cache_dir, season);
        if let Some(cache) = self.load_cache(&cache_path) {
            if !cache.is_stale(6) {
                println!("Using cached data from {} ({} games)", cache.last_updated, cache.games.len());
                return Ok(cache.games);
            }
            println!("Cache is stale, refreshing...");
        }

        // Fetch fresh data
        let games = self.fetch_games_for_range(start_date, end_date)?;

        // Save to cache
        self.save_cache(&cache_path, season, &games)?;

        Ok(games)
    }

    /// Load cached games from file
    fn load_cache(&self, path: &str) -> Option<GameCache> {
        if Path::new(path).exists() {
            let content = fs::read_to_string(path).ok()?;
            serde_json::from_str(&content).ok()
        } else {
            None
        }
    }

    /// Save games to cache file
    fn save_cache(&self, path: &str, season: &str, games: &[GameResult]) -> Result<(), String> {
        let cache = GameCache::new(season.to_string(), games.to_vec());
        let json = serde_json::to_string_pretty(&cache).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Build team mapping from game results
    pub fn build_team_map(games: &[GameResult]) -> HashMap<String, TeamInfo> {
        let mut teams = HashMap::new();

        for game in games {
            if !teams.contains_key(&game.home_team_id) {
                teams.insert(
                    game.home_team_id.clone(),
                    TeamInfo::new(
                        game.home_team_id.clone(),
                        game.home_team_name.clone(),
                        String::new(),
                    ),
                );
            }
            if !teams.contains_key(&game.away_team_id) {
                teams.insert(
                    game.away_team_id.clone(),
                    TeamInfo::new(
                        game.away_team_id.clone(),
                        game.away_team_name.clone(),
                        String::new(),
                    ),
                );
            }
        }

        teams
    }
}

/// Get the current college basketball season string
pub fn current_season() -> String {
    let now = chrono::Local::now();
    let year = now.year();
    let month = now.month();

    // If we're in Jan-April, we're in the second half of the season
    if month <= 6 {
        format!("{}-{}", year - 1, year)
    } else {
        format!("{}-{}", year, year + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_season() {
        let season = current_season();
        assert!(season.contains('-'));
        let parts: Vec<&str> = season.split('-').collect();
        assert_eq!(parts.len(), 2);
    }
}
