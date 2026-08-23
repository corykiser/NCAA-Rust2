// This module handles fetching game data from ESPN and NCAA APIs
// to calculate ELO ratings for NCAA basketball teams

use crate::game_result::{BracketTeam, GameCache, GameResult, TeamInfo};
use crate::ncaa_bracket::{self, BracketField};
use crate::torvik;
use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;

/// Cached bracket data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BracketCache {
    pub tournament_year: i32,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub teams: Vec<BracketTeam>,
    /// Absent in caches written before the Final Four pairing was recorded.
    #[serde(default)]
    pub region_layout: Option<[String; 4]>,
}

impl BracketCache {
    pub fn new(
        tournament_year: i32,
        teams: Vec<BracketTeam>,
        region_layout: [String; 4],
    ) -> Self {
        BracketCache {
            tournament_year,
            last_updated: chrono::Utc::now(),
            teams,
            region_layout: Some(region_layout),
        }
    }
}

/// Data source for fetching game results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataSource {
    ESPN,
    NCAA,
    /// BartTorvik's season game log — one request for the whole season.
    Torvik,
}

/// ESPN API base URL
const ESPN_BASE_URL: &str = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball";

/// NCAA API base URL (henrygd)
const NCAA_BASE_URL: &str = "https://ncaa-api.henrygd.me";

/// Consecutive failed days from the very start of a season that mean the
/// source is unavailable rather than the schedule being empty.
const GIVE_UP_AFTER_CONSECUTIVE_FAILURES: usize = 8;

/// Number of times a transient fetch is retried before the day is failed.
const FETCH_ATTEMPTS: u32 = 3;

/// Rate limit delay in milliseconds (for NCAA API: 5 req/sec max)
const RATE_LIMIT_DELAY_MS: u64 = 250;

/// API client for fetching NCAA basketball data
pub struct ApiClient {
    client: reqwest::blocking::Client,
    source: DataSource,
    cache_dir: String,
    /// Accept and cache a season whose fetch had gaps. Off by default: a season
    /// missing days produces ratings that look normal and are quietly wrong.
    allow_partial: bool,
}

impl ApiClient {
    pub fn new(source: DataSource, cache_dir: &str, allow_partial: bool) -> Self {
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
            allow_partial,
        }
    }

    /// Fetch all teams from ESPN API
    pub fn fetch_all_teams(&self) -> Result<Vec<TeamInfo>, String> {
        match self.source {
            DataSource::ESPN => self.fetch_espn_teams(),
            DataSource::NCAA => self.fetch_ncaa_teams(),
            // Torvik's game log names every team it reports; there is no
            // separate roster endpoint to ask.
            DataSource::Torvik => Err(
                "the barttorvik source has no team endpoint — team names come \
                 from the game log"
                    .to_string(),
            ),
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
            DataSource::Torvik => Err(
                "the barttorvik source fetches a whole season at once; \
                 there is no per-date endpoint"
                    .to_string(),
            ),
        }
    }

    /// GET a URL and parse it as JSON, retrying transient failures.
    ///
    /// A non-2xx response used to be handed straight to `.json()`, so an
    /// upstream `403` surfaced as "expected value at line 1 column 1" — a
    /// parse error for an HTML error page, 158 times over, with the actual
    /// cause nowhere in the output.
    fn get_json(&self, url: &str) -> Result<Value, String> {
        let mut last_error = String::new();

        for attempt in 0..FETCH_ATTEMPTS {
            if attempt > 0 {
                thread::sleep(Duration::from_millis(500 * (1 << attempt)));
            }

            let response = match self.client.get(url).send() {
                Ok(response) => response,
                Err(e) => {
                    last_error = e.to_string();
                    continue;
                }
            };

            let status = response.status();
            if !status.is_success() {
                last_error = match status.as_u16() {
                    403 => format!(
                        "HTTP 403 from {} — the host is refusing this client. \
                         Cloud and datacenter IPs are commonly blocked here; \
                         try `--source torvik`",
                        host_of(url)
                    ),
                    429 => format!("HTTP 429 from {} — rate limited", host_of(url)),
                    _ => format!("HTTP {} from {}", status, host_of(url)),
                };
                // A refusal is not transient; retrying just multiplies it.
                if status.as_u16() == 403 || status.as_u16() == 404 {
                    return Err(last_error);
                }
                continue;
            }

            match response.json::<Value>() {
                Ok(json) => return Ok(json),
                Err(e) => last_error = format!("malformed JSON from {}: {}", host_of(url), e),
            }
        }

        Err(last_error)
    }

    /// GET a URL as text, retrying transient failures.
    fn get_text(&self, url: &str) -> Result<String, String> {
        let mut last_error = String::new();

        for attempt in 0..FETCH_ATTEMPTS {
            if attempt > 0 {
                thread::sleep(Duration::from_millis(500 * (1 << attempt)));
            }

            let response = match self.client.get(url).send() {
                Ok(response) => response,
                Err(e) => {
                    last_error = e.to_string();
                    continue;
                }
            };

            let status = response.status();
            if !status.is_success() {
                last_error = format!("HTTP {} from {}", status, host_of(url));
                if status.as_u16() == 403 || status.as_u16() == 404 {
                    return Err(last_error);
                }
                continue;
            }

            match response.text() {
                Ok(text) => return Ok(text),
                Err(e) => last_error = format!("could not read {}: {}", host_of(url), e),
            }
        }

        Err(last_error)
    }

    /// Fetch games from ESPN for a specific date
    fn fetch_espn_games_for_date(&self, date: NaiveDate) -> Result<Vec<GameResult>, String> {
        let date_str = date.format("%Y%m%d").to_string();
        let url = format!(
            "{}/scoreboard?dates={}&groups=50&limit=500",
            ESPN_BASE_URL, date_str
        );

        let json = self.get_json(&url)?;

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
    ///
    /// The path segments must be zero-padded: `/2026/03/19`, not `/2026/3/19`.
    /// An unpadded path is not an error — it returns `200` with an empty game
    /// list, so every day of the season "succeeded" with nothing in it and the
    /// season came back as zero games with no failure to report.
    fn fetch_ncaa_games_for_date(&self, date: NaiveDate) -> Result<Vec<GameResult>, String> {
        let url = format!(
            "{}/scoreboard/basketball-men/d1/{}",
            NCAA_BASE_URL,
            date.format("%Y/%m/%d")
        );

        let json = self.get_json(&url)?;

        let game_arr = json
            .get("games")
            .and_then(|g| g.as_array())
            .ok_or_else(|| format!("no 'games' array in the response for {}", date))?;

        let mut games = Vec::new();
        for game_data in game_arr {
            let game_obj = match game_data.get("game") {
                Some(game_obj) => game_obj,
                None => continue,
            };
            if let Some(game) = self.parse_ncaa_game(game_obj, date) {
                games.push(game);
            }
        }

        // Rate limiting
        thread::sleep(Duration::from_millis(RATE_LIMIT_DELAY_MS));

        Ok(games)
    }

    /// Parse a single NCAA API game into a GameResult
    fn parse_ncaa_game(&self, game: &Value, date: NaiveDate) -> Option<GameResult> {
        parse_ncaa_game(game, date)
    }

    /// Fetch all games for a date range, reporting which days failed.
    ///
    /// Failures used to be printed as warnings and then forgotten, and the
    /// truncated result was cached as though it were the complete season.
    pub fn fetch_games_for_range(
        &self,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> (Vec<GameResult>, Vec<(NaiveDate, String)>) {
        let mut all_games = Vec::new();
        let mut failures = Vec::new();
        let mut current_date = start_date;
        let total_days = (end_date - start_date).num_days() + 1;
        let mut days_processed = 0;

        println!("Fetching games from {} to {}...", start_date, end_date);

        while current_date <= end_date {
            match self.fetch_games_for_date(current_date) {
                Ok(games) => {
                    if !games.is_empty() {
                        print!(".");
                    }
                    all_games.extend(games);
                }
                Err(e) => failures.push((current_date, e)),
            }

            // A source that is refusing us refuses every day the same way.
            // Walking the whole season to say so costs minutes and buries the
            // reason under 158 copies of itself.
            if failures.len() >= GIVE_UP_AFTER_CONSECUTIVE_FAILURES
                && failures.len() as i64 == days_processed + 1
            {
                println!();
                eprintln!(
                    "Giving up after {} days in a row failed — the source is not \
                     answering, not just missing a day.",
                    failures.len()
                );
                return (all_games, failures);
            }

            days_processed += 1;
            if days_processed % 30 == 0 {
                println!(" [{}/{}]", days_processed, total_days);
            }

            current_date = current_date.succ_opt().unwrap();
        }

        println!("\nFetched {} games total", all_games.len());
        (all_games, failures)
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

        if self.source == DataSource::Torvik {
            return self.fetch_season_torvik(season, end_year);
        }

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
        let (games, failures) = self.fetch_games_for_range(start_date, end_date);

        if !failures.is_empty() {
            let sample: Vec<String> = failures
                .iter()
                .take(5)
                .map(|(date, err)| format!("  {}: {}", date, err))
                .collect();
            let message = format!(
                "{} of {} days failed to fetch:\n{}{}",
                failures.len(),
                (end_date - start_date).num_days() + 1,
                sample.join("\n"),
                if failures.len() > sample.len() {
                    format!("\n  ... and {} more", failures.len() - sample.len())
                } else {
                    String::new()
                }
            );

            if !self.allow_partial {
                return Err(format!(
                    "{}\nRatings from an incomplete season look normal and are wrong. \
                     Retry, or pass --allow-partial-data to accept the gaps.",
                    message
                ));
            }
            eprintln!("Warning: {}", message);
            eprintln!("Proceeding with incomplete data (--allow-partial-data). Not caching.");
            return Ok(games);
        }

        if games.is_empty() {
            return Err(format!(
                "no games returned for season {} ({} to {})",
                season, start_date, end_date
            ));
        }

        // Only a complete fetch is worth caching; a partial one would be served
        // back as authoritative for the rest of the staleness window.
        self.save_cache(&cache_path, season, &games)?;

        Ok(games)
    }

    /// Fetch a whole season from BartTorvik in one request.
    ///
    /// `end_year` is the season's ending calendar year — 2027 for 2026-27,
    /// which is how Torvik labels a season.
    fn fetch_season_torvik(&self, season: &str, end_year: i32) -> Result<Vec<GameResult>, String> {
        let cache_path = format!("{}/games_{}.json", self.cache_dir, season);
        if let Some(cache) = self.load_cache(&cache_path) {
            if !cache.is_stale(6) {
                println!(
                    "Using cached data from {} ({} games)",
                    cache.last_updated,
                    cache.games.len()
                );
                return Ok(cache.games);
            }
            println!("Cache is stale, refreshing...");
        }

        let url = torvik::season_url(end_year);
        println!("Fetching the {} season game log from barttorvik.com...", season);
        let csv_text = self
            .get_text(&url)
            .map_err(|e| format!("barttorvik game log for {}: {}", season, e))?;

        if csv_text.trim().is_empty() {
            return Err(format!(
                "barttorvik has no games for {} yet. The file fills in as the \
                 season is played, so this is what an unplayed season looks \
                 like rather than an outage.",
                season
            ));
        }

        let games = torvik::parse_game_log(&csv_text)?;
        println!("Fetched {} games total", games.len());

        // An in-progress season is complete by definition — there is no set of
        // days that failed, only games that have not been played yet.
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

    /// Fetch the tournament field for a year from the NCAA's published bracket.
    ///
    /// `tournament_year` is the year the tournament ends: 2027 for March
    /// Madness 2027. Independent of `--source`, which only decides where the
    /// ratings come from.
    pub fn fetch_tournament_bracket(&self, tournament_year: i32) -> Result<BracketField, String> {
        let cache_path = format!("{}/bracket_{}.json", self.cache_dir, tournament_year);
        if let Some(cache) = self.load_bracket_cache(&cache_path) {
            if cache.teams.len() == 64 && cache.region_layout.is_some() {
                println!(
                    "Using cached bracket from {} ({} teams)",
                    cache.last_updated,
                    cache.teams.len()
                );
                return Ok(BracketField {
                    teams: cache.teams,
                    region_layout: cache.region_layout.expect("checked above"),
                });
            }
            // A cache from before the bracket was complete, or from the old
            // scoreboard-scraping path, which had no region layout in it.
            println!("Cached bracket is incomplete, refetching...");
        }

        println!(
            "Fetching the {} bracket from the NCAA...",
            tournament_year
        );
        let json = self.get_json(&ncaa_bracket::bracket_url(tournament_year))?;
        let field = ncaa_bracket::parse_bracket(&json, tournament_year)?;

        println!(
            "Found {} tournament teams; {} plays {} and {} plays {} in the Final Four",
            field.teams.len(),
            field.region_layout[0],
            field.region_layout[1],
            field.region_layout[2],
            field.region_layout[3],
        );

        self.save_bracket_cache(&cache_path, tournament_year, &field)?;
        Ok(field)
    }

    /// Load cached bracket from file
    fn load_bracket_cache(&self, path: &str) -> Option<BracketCache> {
        if Path::new(path).exists() {
            let content = fs::read_to_string(path).ok()?;
            serde_json::from_str(&content).ok()
        } else {
            None
        }
    }

    /// Save bracket to cache file
    fn save_bracket_cache(&self, path: &str, year: i32, field: &BracketField) -> Result<(), String> {
        let cache = BracketCache::new(year, field.teams.clone(), field.region_layout.clone());
        let json = serde_json::to_string_pretty(&cache).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Load bracket teams from a local JSON file
/// File format: array of objects with team_id, team_name, seed, region
pub fn load_bracket_from_file(path: &str) -> Result<Vec<BracketTeam>, String> {
    if !Path::new(path).exists() {
        return Err(format!("Bracket file not found: {}", path));
    }

    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;

    // Try to parse as BracketCache first (our cached format)
    if let Ok(cache) = serde_json::from_str::<BracketCache>(&content) {
        return Ok(cache.teams);
    }

    // Try to parse as raw array of BracketTeam
    if let Ok(teams) = serde_json::from_str::<Vec<BracketTeam>>(&content) {
        return Ok(teams);
    }

    Err("Could not parse bracket file. Expected JSON array of teams with team_id, team_name, seed, region".to_string())
}

/// Pull a usable team name out of an NCAA API `names` object.
///
/// `names.full` is present on every game and **empty on most of them** — the
/// feed only fills it in for a subset of schools. Reading it alone produced
/// games between two teams named "", which ELO happily merged into a single
/// phantom team with a few thousand games.
fn ncaa_team_name(names_obj: &Value) -> Option<String> {
    for key in ["short", "full", "seo", "char6"] {
        if let Some(name) = names_obj.get(key).and_then(|v| v.as_str()) {
            let name = name.trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Parse one NCAA API scoreboard game. Free-standing so it can be tested
/// against a recorded payload without a client.
fn parse_ncaa_game(game: &Value, date: NaiveDate) -> Option<GameResult> {
    let game_id = game.get("gameID")?.as_str()?.to_string();

    if game.get("gameState")?.as_str()? != "final" {
        return None;
    }

    let home = game.get("home")?;
    let away = game.get("away")?;
    let home_team_name = ncaa_team_name(home.get("names")?)?;
    let away_team_name = ncaa_team_name(away.get("names")?)?;
    let home_score: u32 = home.get("score")?.as_str()?.trim().parse().ok()?;
    let away_score: u32 = away.get("score")?.as_str()?.trim().parse().ok()?;

    let conference_of = |side: &Value| -> Option<String> {
        side.get("conferences")?
            .as_array()?
            .first()?
            .get("conferenceSeo")?
            .as_str()
            .map(|c| c.to_string())
    };
    let is_conference_game = match (conference_of(home), conference_of(away)) {
        (Some(a), Some(b)) if !a.is_empty() => a == b,
        _ => false,
    };

    Some(GameResult {
        game_id,
        date,
        // Normalized names as ids: the feed has no stable numeric team id, and
        // `seo` disagrees with itself across seasons.
        home_team_id: crate::names::normalize(&home_team_name),
        home_team_name,
        away_team_id: crate::names::normalize(&away_team_name),
        away_team_name,
        home_score,
        away_score,
        // The scoreboard feed does not mark neutral sites.
        is_neutral_site: false,
        is_conference_game,
        is_completed: true,
    })
}

/// Host portion of a URL, for error messages that name what refused us.
fn host_of(url: &str) -> &str {
    url.split("://")
        .nth(1)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or(url)
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

    /// The scoreboard feed leaves `names.full` empty for most schools, so the
    /// parser has to fall back to `short`. Reading `full` alone produced games
    /// between two teams called "".
    #[test]
    fn ncaa_games_are_named_even_when_full_is_empty() {
        let game = serde_json::json!({
            "gameID": "6349566",
            "gameState": "final",
            "home": {
                "score": "64",
                "names": {"char6": "TENN", "short": "Tennessee", "seo": "tennessee", "full": ""},
                "conferences": [{"conferenceName": "", "conferenceSeo": "sec"}]
            },
            "away": {
                "score": "44",
                "names": {"char6": "FLA", "short": "Florida", "seo": "florida", "full": ""},
                "conferences": [{"conferenceName": "", "conferenceSeo": "sec"}]
            }
        });
        let date = NaiveDate::from_ymd_opt(2025, 2, 1).unwrap();
        let parsed = parse_ncaa_game(&game, date).expect("parses");

        assert_eq!(parsed.home_team_name, "Tennessee");
        assert_eq!(parsed.away_team_name, "Florida");
        assert_eq!(parsed.home_score, 64);
        assert!(parsed.is_conference_game, "both teams are in the SEC");
        assert_ne!(parsed.home_team_id, parsed.away_team_id);
    }

    #[test]
    fn ncaa_games_that_are_not_final_are_skipped() {
        let game = serde_json::json!({
            "gameID": "1",
            "gameState": "live",
            "home": {"score": "10", "names": {"short": "A"}},
            "away": {"score": "8", "names": {"short": "B"}}
        });
        let date = NaiveDate::from_ymd_opt(2025, 2, 1).unwrap();
        assert!(parse_ncaa_game(&game, date).is_none());
    }

    /// The path segments are zero-padded. An unpadded one returns 200 with no
    /// games, which reads as a season in which nothing was played.
    #[test]
    fn ncaa_scoreboard_dates_are_zero_padded() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 1).unwrap();
        assert_eq!(date.format("%Y/%m/%d").to_string(), "2026/03/01");
    }

    #[test]
    fn test_current_season() {
        let season = current_season();
        assert!(season.contains('-'));
        let parts: Vec<&str> = season.split('-').collect();
        assert_eq!(parts.len(), 2);
    }
}
