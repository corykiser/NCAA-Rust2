// This file is used to ingest the data from the csv file that contains the 538 ratings and store it in a struct to be used later in other parts of the program
// Also supports creating tournament info from custom ELO ratings calculated from live API data

use serde::{Serialize, Deserialize};
use csv;
use csv::StringRecord;
use std::collections::HashMap;
use std::sync::Arc;
use crate::elo::EloSystem;
use crate::game_result::BracketTeam;

/// Atomically reference-counted Team for efficient sharing without cloning.
/// Arc is used instead of Rc because it's thread-safe for parallel processing with rayon.
pub type RcTeam = Arc<Team>;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Team {
    pub name: String,
    pub seed: i32,
    pub region: String,
    pub rating: f32,
    /// Index into tournament.teams array (0-63) for fast lookup
    #[serde(default)]
    pub team_index: u8,
}

impl Team {
    pub fn new(name: String, seed: i32, region: String, rating: f32) -> Team {
        Team {
            name,
            seed,
            region,
            rating,
            team_index: 0, // Will be set when added to tournament
        }
    }

    pub fn with_index(name: String, seed: i32, region: String, rating: f32, team_index: u8) -> Team {
        Team {
            name,
            seed,
            region,
            rating,
            team_index,
        }
    }
    pub fn print(&self) {
        println!("{:?}", self);
    }

}
impl PartialEq for Team {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// Pre-computed win probabilities for all team pairs
/// Avoids expensive powf() calls during bracket creation
#[derive(Debug, Clone)]
pub struct ProbabilityCache {
    /// probs[team_a_idx][team_b_idx] = probability that team_a beats team_b
    probs: [[f64; 64]; 64],
}

impl ProbabilityCache {
    /// Create cache from team ratings
    pub fn new(teams: &[RcTeam]) -> Self {
        let mut probs = [[0.5f64; 64]; 64];

        for (i, team_a) in teams.iter().enumerate() {
            for (j, team_b) in teams.iter().enumerate() {
                if i != j {
                    let rating_diff = team_a.rating as f64 - team_b.rating as f64;
                    probs[i][j] = 1.0 / (1.0 + 10.0f64.powf(-rating_diff * 30.464 / 400.0));
                } else {
                    probs[i][j] = 0.5; // Same team
                }
            }
        }

        ProbabilityCache { probs }
    }

    /// Get win probability for team_a vs team_b (using team indices)
    #[inline(always)]
    pub fn get(&self, team_a_idx: u8, team_b_idx: u8) -> f64 {
        unsafe {
            *self.probs.get_unchecked(team_a_idx as usize).get_unchecked(team_b_idx as usize)
        }
    }
}

#[derive(Debug)]
pub struct TournamentInfo {
    pub teams: Vec<RcTeam>,
    pub round1: [[i32; 2]; 8],
    pub round2: [[i32; 4]; 4],
    pub round3: [[i32; 8]; 2],
    pub round4: [[i32; 16]; 1],
    pub regions: Vec<Vec<RcTeam>>,
    /// Fast lookup map: (region, seed) -> RcTeam
    /// This avoids O(n) filtering in bracket construction
    pub team_lookup: HashMap<(String, i32), RcTeam>,
    /// Pre-computed win probabilities for all team pairs
    pub prob_cache: ProbabilityCache,
}

impl TournamentInfo {
    // Inititialize the vector of teams to be hold 64 teams
    pub fn initialize(file_path: &str) -> TournamentInfo {
        //structure of tournament: What seed plays the other seeds in each round?
        let round1: [[i32; 2]; 8] = [
            [1, 16],
            [2, 15],
            [3, 14],
            [4, 13],
            [5, 12],
            [6, 11],
            [7, 10],
            [8, 9],
        ];
        let round2: [[i32; 4]; 4] = [
            [1, 16, 8, 9],
            [5, 12, 4, 13],
            [6, 11, 3, 14],
            [7, 10, 2, 15],
        ];
        let round3: [[i32; 8]; 2] = [[1, 16, 8, 9, 5, 12, 4, 13], [6, 11, 3, 14, 7, 10, 2, 15]];
        let round4: [[i32; 16]; 1] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]];

        let mut teams: Vec<RcTeam> = Vec::with_capacity(64);
        let mut team_lookup: HashMap<(String, i32), RcTeam> = HashMap::with_capacity(64);

        //TODO check if file exists and download if it doesn't exist OR specify file path as an argument
        //let file_path = "/Users/corydkiser/Documents/ncaa/fivethirtyeight_ncaa_forecasts.csv";
        let mut rdr = csv::Reader::from_path(file_path).expect("file access error");
        let mut mensrecords: Vec<StringRecord> = Vec::new(); //holds the csv records

        // Loop over each record.
        for result in rdr.records() {
            let record = &result.unwrap();
            if record[0].starts_with("mens")
                && record[1].contains("2023-03-15")
                //TODO find the latest date with the last exactly 64 teams (there are 64 record[3] entries that equal 1) to sort out the play in games
                && record[3].contains("1.0")
            {
                mensrecords.push(record.clone());
            }
        }

        //create containers
        let mut rating: [f64; 64] = [0.0; 64];
        let mut teamid: [u32; 64] = [0; 64];
        let mut seed: [u32; 64] = [0; 64];
        let mut name: Vec<String> = Vec::new();
        let mut region: Vec<String> = Vec::new();

        for i in 0..mensrecords.len() {
            rating[i] = mensrecords[i][14].parse().unwrap(); //populate Team Ratings array
            teamid[i] = mensrecords[i][12].parse().unwrap(); //populate team id array
            name.push(mensrecords[i][13].to_string()); //populate team name array
            region.push(mensrecords[i][15].to_string()); //populate regions array
            //below removes non ascii digits "a" and "b" from team seeds
            //let test = "12b3as>";
            //let test2: String = test.to_string().chars().filter(|x| x.is_ascii_digit()).collect();
            if mensrecords[i][16].ends_with("a") || mensrecords[i][16].ends_with("b") {
                //let length = mensrecords[i][16].len();
                let mut tempstring = mensrecords[i][16].to_string();
                tempstring.pop();
                seed[i] = tempstring.parse().unwrap(); //populate team seed array
            } else {
                seed[i] = mensrecords[i][16].parse().unwrap();
            }
            //add team to the vector as Arc<Team> with team_index
            let team = Arc::new(Team::with_index(
                name[i].clone(),
                seed[i] as i32,
                region[i].clone(),
                rating[i] as f32,
                i as u8,  // team_index
            ));
            team_lookup.insert((region[i].clone(), seed[i] as i32), Arc::clone(&team));
            teams.push(team);
        }
        assert!(teams.len() == 64, "There are not 64 teams in the tournament");

        // Create probability cache from teams
        let prob_cache = ProbabilityCache::new(&teams);

        // Build region vectors using Arc::clone (cheap atomic reference counting, no data copy)
        let east: Vec<RcTeam> = teams.iter()
            .filter(|x| x.region == "East")
            .map(Arc::clone)
            .collect();

        let west: Vec<RcTeam> = teams.iter()
            .filter(|x| x.region == "West")
            .map(Arc::clone)
            .collect();

        let south: Vec<RcTeam> = teams.iter()
            .filter(|x| x.region == "South")
            .map(Arc::clone)
            .collect();

        let midwest: Vec<RcTeam> = teams.iter()
            .filter(|x| x.region == "Midwest")
            .map(Arc::clone)
            .collect();

        let regions = vec![east, west, south, midwest];

        TournamentInfo {
            teams,
            round1,
            round2,
            round3,
            round4,
            regions,
            team_lookup,
            prob_cache,
        }
    }
    pub fn print(&self) {
        println!("{:?}", self);
    }

    /// Get a team by region and seed using O(1) lookup
    /// Returns an Arc clone (cheap atomic reference count increment)
    #[inline]
    pub fn get_team(&self, region: &str, seed: i32) -> RcTeam {
        Arc::clone(self.team_lookup.get(&(region.to_string(), seed))
            .unwrap_or_else(|| panic!("Team not found: region={}, seed={}", region, seed)))
    }

    /// Create TournamentInfo from ELO ratings and bracket team information
    /// This allows using custom-calculated ELO ratings from live API data
    /// instead of the FiveThirtyEight CSV file
    pub fn from_elo_ratings(elo_system: &EloSystem, bracket_teams: Vec<BracketTeam>) -> TournamentInfo {
        // Tournament structure (same as initialize)
        let round1: [[i32; 2]; 8] = [
            [1, 16], [2, 15], [3, 14], [4, 13],
            [5, 12], [6, 11], [7, 10], [8, 9],
        ];
        let round2: [[i32; 4]; 4] = [
            [1, 16, 8, 9], [5, 12, 4, 13],
            [6, 11, 3, 14], [7, 10, 2, 15],
        ];
        let round3: [[i32; 8]; 2] = [
            [1, 16, 8, 9, 5, 12, 4, 13],
            [6, 11, 3, 14, 7, 10, 2, 15],
        ];
        let round4: [[i32; 16]; 1] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]];

        let mut teams: Vec<RcTeam> = Vec::with_capacity(64);
        let mut team_lookup: HashMap<(String, i32), RcTeam> = HashMap::with_capacity(64);

        // Convert bracket teams to Team structs with ELO ratings
        for (idx, bracket_team) in bracket_teams.iter().enumerate() {
            // Look up the team's ELO rating
            let rating = if let Some(elo_rating) = elo_system.find_team_by_name(&bracket_team.team_name) {
                elo_system.to_538_scale(&elo_rating.team_id)
            } else {
                // If team not found in ELO system, use a default rating
                // This might happen for play-in game teams with limited data
                println!("Warning: No ELO rating found for '{}', using default", bracket_team.team_name);
                75.0 // Middle-of-the-road default
            };

            let team = Arc::new(Team::with_index(
                bracket_team.team_name.clone(),
                bracket_team.seed,
                bracket_team.region.clone(),
                rating,
                idx as u8,  // team_index
            ));
            team_lookup.insert((bracket_team.region.clone(), bracket_team.seed), Arc::clone(&team));
            teams.push(team);
        }

        assert!(teams.len() == 64, "There must be exactly 64 teams in the tournament");

        // Create probability cache from teams
        let prob_cache = ProbabilityCache::new(&teams);

        // Organize teams by region using Arc::clone (cheap atomic reference counting)
        let east: Vec<RcTeam> = teams.iter().filter(|x| x.region == "East").map(Arc::clone).collect();
        let west: Vec<RcTeam> = teams.iter().filter(|x| x.region == "West").map(Arc::clone).collect();
        let south: Vec<RcTeam> = teams.iter().filter(|x| x.region == "South").map(Arc::clone).collect();
        let midwest: Vec<RcTeam> = teams.iter().filter(|x| x.region == "Midwest").map(Arc::clone).collect();

        let regions = vec![east, west, south, midwest];

        TournamentInfo {
            teams,
            round1,
            round2,
            round3,
            round4,
            regions,
            team_lookup,
            prob_cache,
        }
    }

    /// Create a sample bracket team list for testing
    /// In production, this would come from the NCAA bracket announcement
    pub fn sample_bracket_teams() -> Vec<BracketTeam> {
        // This is a sample based on 2024 tournament structure
        // Would need to be updated each year when brackets are announced
        let regions = ["East", "West", "South", "Midwest"];
        let mut teams = Vec::new();

        // Sample teams - in reality, these would come from the official bracket
        let sample_teams_by_region = [
            // East (sample teams)
            vec![
                ("Connecticut", 1), ("Iowa State", 2), ("Illinois", 3), ("Auburn", 4),
                ("San Diego State", 5), ("BYU", 6), ("Texas", 7), ("Florida Atlantic", 8),
                ("Northwestern", 9), ("Drake", 10), ("Duquesne", 11), ("UAB", 12),
                ("Yale", 13), ("Morehead State", 14), ("Long Beach State", 15), ("Stetson", 16),
            ],
            // West
            vec![
                ("North Carolina", 1), ("Arizona", 2), ("Baylor", 3), ("Alabama", 4),
                ("Saint Mary's", 5), ("Clemson", 6), ("Dayton", 7), ("Mississippi State", 8),
                ("Michigan State", 9), ("Nevada", 10), ("New Mexico", 11), ("Grand Canyon", 12),
                ("Charleston", 13), ("Colgate", 14), ("Long Island", 15), ("Wagner", 16),
            ],
            // South
            vec![
                ("Houston", 1), ("Marquette", 2), ("Kentucky", 3), ("Duke", 4),
                ("Wisconsin", 5), ("Texas Tech", 6), ("Florida", 7), ("Nebraska", 8),
                ("Texas A&M", 9), ("Colorado", 10), ("NC State", 11), ("James Madison", 12),
                ("Vermont", 13), ("Oakland", 14), ("Western Kentucky", 15), ("Longwood", 16),
            ],
            // Midwest
            vec![
                ("Purdue", 1), ("Tennessee", 2), ("Creighton", 3), ("Kansas", 4),
                ("Gonzaga", 5), ("South Carolina", 6), ("Texas", 7), ("Utah State", 8),
                ("TCU", 9), ("Colorado State", 10), ("Oregon", 11), ("McNeese", 12),
                ("Samford", 13), ("Akron", 14), ("Grambling State", 15), ("Montana State", 16),
            ],
        ];

        for (region_idx, region_teams) in sample_teams_by_region.iter().enumerate() {
            for (name, seed) in region_teams {
                teams.push(BracketTeam::new(
                    name.to_lowercase().replace(' ', "-"),
                    name.to_string(),
                    *seed,
                    regions[region_idx].to_string(),
                ));
            }
        }

        teams
    }
}