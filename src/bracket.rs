use rand::Rng;
use std::sync::Arc;
use crate::ingest::{Team, RcTeam, TournamentInfo};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeedScoring {
    Add,
    Multiply,
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct ScoringConfig {
    pub round_scores: [f64; 6],
    pub round_seed_scoring: [SeedScoring; 6],
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            round_scores: [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            round_seed_scoring: [
                SeedScoring::Add,      // R1: 1 + seed
                SeedScoring::Add,      // R2: 2 + seed
                SeedScoring::Add,      // R3: 4 + seed
                SeedScoring::Multiply, // R4: 8 * seed
                SeedScoring::Multiply, // R5: 16 * seed
                SeedScoring::Multiply, // R6: 32 * seed
            ],
        }
    }
}

/// Pre-computed score lookup table for fast scoring
/// Avoids repeated calculate_win_score() calls (was 18.9 billion calls per run)
#[derive(Debug, Clone, Copy)]
pub struct ScoreTable {
    /// scores[round][seed] = points for correct pick
    /// Seeds 1-16, index 0 unused for cleaner indexing
    scores: [[f64; 17]; 6],
}

impl ScoreTable {
    /// Create a new score table from a scoring config
    pub fn new(config: &ScoringConfig) -> Self {
        let mut scores = [[0.0; 17]; 6];

        for round in 0..6 {
            let base_score = config.round_scores[round];
            for seed in 1..=16 {
                scores[round][seed] = match config.round_seed_scoring[round] {
                    SeedScoring::Add => base_score + seed as f64,
                    SeedScoring::Multiply => base_score * seed as f64,
                    SeedScoring::None => base_score,
                };
            }
        }

        ScoreTable { scores }
    }

    /// Get score for a correct pick (inlined for performance)
    #[inline(always)]
    pub fn get(&self, round: usize, seed: i32) -> f64 {
        // Safety: round is always 0-5, seed is always 1-16
        unsafe {
            *self.scores.get_unchecked(round).get_unchecked(seed as usize)
        }
    }
}

impl Default for ScoreTable {
    fn default() -> Self {
        Self::new(&ScoringConfig::default())
    }
}

/// Game struct uses reference-counted Team pointers (RcTeam) to avoid
/// cloning Team data. Arc::clone() is O(1) - just increments a counter.
#[derive(Debug, Clone)]
pub struct Game {
    pub team1: RcTeam,
    pub team2: RcTeam,
    pub team1prob: f64,
    pub team2prob: f64,
    pub winnerprob: f64,
    pub winner: RcTeam,
    pub hilo: bool, //did the lower seed (or the region first in alphabetical order win) win?
}


// Using RcTeam (Rc<Team>) avoids cloning Team data.
// Arc::clone() is O(1) - just increments a reference counter.
impl Game {
    /// Simulate a game between two teams.
    /// Takes RcTeam references and uses Arc::clone() which is O(1).
    pub fn new(team1: &RcTeam, team2: &RcTeam) -> Game {
        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let mut rng = rand::thread_rng();
        let rand_num: f64 = rng.gen(); // generates a float between 0 and 1

        // Arc::clone is O(1) - just increments reference count
        let (winner, winnerprob) = if rand_num < team1prob {
            (Arc::clone(team1), team1prob)
        } else {
            (Arc::clone(team2), team2prob)
        };

        // Check if the regions are the same. If so, did the lower seed win? If not, did the region first in alphabetical order win?
        let hilo: bool = if team1.region == team2.region {
            team1.seed < team2.seed
        } else {
            team1.region < team2.region
        };

        Game {
            team1: Arc::clone(team1),
            team2: Arc::clone(team2),
            team1prob,
            team2prob,
            winnerprob,
            winner,
            hilo,
        }
    }
    //TODO create a new_from_partial_binary() version of this that takes in a partial bracket and fills in the rest of the bracket
    //You can use Some(hilo) and None in Vec<Option<bool>> to indicate if what parts need to be filled in
    //This would be useful for filling in the first round.
    /// Create a game from binary (hilo) representation.
    /// Uses RcTeam to avoid cloning Team data - Arc::clone() is O(1).
    pub fn new_from_binary(team1: &RcTeam, team2: &RcTeam, hilo: bool) -> Game {
        // Determine winner based on hilo flag
        // Using references to avoid any cloning until we need to store
        let (low_seed_team, high_seed_team) = if team1.seed < team2.seed {
            (team1, team2)
        } else {
            (team2, team1)
        };

        let (low_alpha_team, high_alpha_team) = if team1.region < team2.region {
            (team1, team2)
        } else {
            (team2, team1)
        };

        // Select winner based on hilo and region match
        let winner: &RcTeam = if team1.region == team2.region {
            if hilo { low_seed_team } else { high_seed_team }
        } else {
            if hilo { low_alpha_team } else { high_alpha_team }
        };

        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let winnerprob = if Arc::ptr_eq(winner, team1) {
            team1prob
        } else {
            team2prob
        };

        Game {
            team1: Arc::clone(team1),
            team2: Arc::clone(team2),
            team1prob,
            team2prob,
            winnerprob,
            winner: Arc::clone(winner),
            hilo,
        }
    }
    pub fn print(&self) {
        println!("{:?}", self);
    }
}

//todo smart mutate function that only mutates the parts of the bracket that are uncertain (i.e. win probability is between 0.4 and 0.6ç)

/// Result of comparing two brackets with weighted distance metric
#[derive(Debug, Clone)]
pub struct BracketDistance {
    /// Total weighted distance (lower = more similar)
    pub total_distance: f64,
    /// Similarity percentage (0.0 to 1.0, higher = more similar)
    pub similarity: f64,
    /// Distance contribution from each round [R1, R2, Sweet16, Elite8, F4, Championship]
    pub round_distances: [f64; 6],
    /// Number of matching picks per round
    pub round_matches: [usize; 6],
    /// Total games per round for reference
    pub round_totals: [usize; 6],
    /// Whether both brackets have the same champion
    pub champion_match: bool,
}

impl BracketDistance {
    /// Pretty print the distance breakdown
    pub fn print(&self) {
        let round_names = ["Round 1", "Round 2", "Sweet 16", "Elite 8", "Final Four", "Championship"];

        println!("\nBracket Similarity Analysis");
        println!("============================");
        println!("Overall Similarity: {:.1}%", self.similarity * 100.0);
        println!("Total Distance: {:.2}", self.total_distance);
        println!("Same Champion: {}", if self.champion_match { "Yes" } else { "No" });
        println!();
        println!("{:<14} {:>8} {:>12}", "Round", "Matches", "Distance");
        println!("{}", "-".repeat(36));

        for i in 0..6 {
            println!(
                "{:<14} {:>3}/{:<4} {:>12.2}",
                round_names[i],
                self.round_matches[i],
                self.round_totals[i],
                self.round_distances[i]
            );
        }
        println!();
    }

    /// Returns true if brackets are identical
    pub fn is_identical(&self) -> bool {
        self.total_distance == 0.0
    }

    /// Returns the percentage of games that match in later rounds (Sweet 16+)
    pub fn late_round_match_pct(&self) -> f64 {
        let late_matches: usize = self.round_matches[2..].iter().sum();
        let late_totals: usize = self.round_totals[2..].iter().sum();
        late_matches as f64 / late_totals as f64
    }
}

#[derive(Debug, Clone)]
pub struct Bracket {
    pub games: Vec<Game>, // Flattened vector of 63 games. 0-31: R1, 32-47: R2, 48-55: R3, 56-59: R4, 60-61: R5, 62: R6
    pub winner: RcTeam, //winner of tournament (reference counted, cheap to clone)
    pub prob: f64,  //probability of bracket scenario occuring
    pub score: f64, //score if bracket were perfectly picked
    pub sim_score: f64, //score from simulations
    pub expected_value: f64, //expected value of bracket
    pub binary: Vec<bool>,
}

impl PartialEq for Bracket {
    fn eq(&self, other: &Self) -> bool {
        self.binary == other.binary
    }
}

impl Bracket{
    /// Helper to calculate score for a game win based on config
    #[inline]
    fn calculate_win_score(round_idx: usize, seed: i32, config: &ScoringConfig) -> f64 {
        let base_score = config.round_scores[round_idx];
        match config.round_seed_scoring[round_idx] {
            SeedScoring::Add => base_score + seed as f64,
            SeedScoring::Multiply => base_score * seed as f64,
            SeedScoring::None => base_score,
        }
    }

    /// Create a new random bracket using Monte Carlo simulation.
    pub fn new(tournamentinfo: &TournamentInfo, config: Option<&ScoringConfig>) -> Bracket{
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);

        // Preallocate vector for all games
        let mut games: Vec<Game> = Vec::with_capacity(63);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;
        let mut binary: Vec<bool> = Vec::with_capacity(63);

        let region_names = ["East", "West", "South", "Midwest"];

        // Matchup indices for direct lookup (avoids search)
        const R2_INDICES: [(usize, usize); 4] = [(0, 7), (4, 3), (5, 2), (6, 1)];
        const R3_INDICES: [(usize, usize); 2] = [(0, 1), (2, 3)];
        const R4_INDICES: [(usize, usize); 1] = [(0, 1)];

        // Round 1 (Index 0)
        for region in &region_names {
            for matchup in tournamentinfo.round1 {
                let team1 = tournamentinfo.get_team(region, matchup[0]);
                let team2 = tournamentinfo.get_team(region, matchup[1]);

                let game = Game::new(&team1, &team2);
                let win_score = Self::calculate_win_score(0, game.winner.seed, config);

                prob *= game.winnerprob;
                score += win_score;
                expected_value += game.winnerprob * win_score;
                binary.push(game.hilo);
                games.push(game);
            }
        }

        // Round 2 (Index 1)
        let r1_start = 0;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 8;
            for &(idx1, idx2) in &R2_INDICES {
                let team1 = Arc::clone(&games[r1_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r1_start + offset + idx2].winner);
                let game = Game::new(&team1, &team2);
                let win_score = Self::calculate_win_score(1, game.winner.seed, config);

                prob *= game.winnerprob;
                score += win_score;
                expected_value += game.winnerprob * win_score;
                binary.push(game.hilo);
                games.push(game);
            }
        }

        // Round 3 (Sweet 16) (Index 2)
        let r2_start = 32;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 4;
            for &(idx1, idx2) in &R3_INDICES {
                let team1 = Arc::clone(&games[r2_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r2_start + offset + idx2].winner);
                let game = Game::new(&team1, &team2);
                let win_score = Self::calculate_win_score(2, game.winner.seed, config);

                prob *= game.winnerprob;
                score += win_score;
                expected_value += game.winnerprob * win_score;
                binary.push(game.hilo);
                games.push(game);
            }
        }

        // Round 4 (Elite 8) (Index 3)
        let r3_start = 48;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 2;
            for &(idx1, idx2) in &R4_INDICES {
                let team1 = Arc::clone(&games[r3_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r3_start + offset + idx2].winner);
                let game = Game::new(&team1, &team2);
                let win_score = Self::calculate_win_score(3, game.winner.seed, config);

                prob *= game.winnerprob;
                score += win_score;
                expected_value += game.winnerprob * win_score;
                binary.push(game.hilo);
                games.push(game);
            }
        }

        // Final Four (Index 4)
        // South vs Midwest
        let r4_start = 56;
        let south_winner = Arc::clone(&games[r4_start + 2].winner);
        let midwest_winner = Arc::clone(&games[r4_start + 3].winner);

        let game = Game::new(&south_winner, &midwest_winner);
        let win_score = Self::calculate_win_score(4, game.winner.seed, config);

        prob *= game.winnerprob;
        score += win_score;
        expected_value += game.winnerprob * win_score;
        binary.push(game.hilo);
        games.push(game);

        // East vs West
        let east_winner = Arc::clone(&games[r4_start + 0].winner);
        let west_winner = Arc::clone(&games[r4_start + 1].winner);

        let game = Game::new(&east_winner, &west_winner);
        // Recalculate win score for the winner of this game
        let win_score = Self::calculate_win_score(4, game.winner.seed, config);

        prob *= game.winnerprob;
        score += win_score;
        expected_value += game.winnerprob * win_score;
        binary.push(game.hilo);
        games.push(game);

        // Championship game (Index 5)
        let r5_start = 60;
        let team1 = Arc::clone(&games[r5_start + 0].winner);
        let team2 = Arc::clone(&games[r5_start + 1].winner);
        let game = Game::new(&team1, &team2);
        let win_score = Self::calculate_win_score(5, game.winner.seed, config);

        prob *= game.winnerprob;
        score += win_score;
        expected_value += game.winnerprob * win_score;

        let tournament_winner = Arc::clone(&game.winner);
        binary.push(game.hilo);
        games.push(game);

        debug_assert!(binary.len() == 63);

        Bracket{
            games,
            winner: tournament_winner,
            prob,
            score,
            sim_score: 0.0,
            expected_value,
            binary,
        }
    }

    pub fn score(&self, referencebracket: &Bracket, config: Option<&ScoringConfig>) -> f64{
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);

        let mut score: f64 = 0.0;

        // Using pointer comparison for performance
        // Round 1
        for i in 0..32{
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += Self::calculate_win_score(0, self.games[i].winner.seed, config);
            }
        }
        // Round 2
        for i in 0..16{
            if Arc::ptr_eq(&self.games[32+i].winner, &referencebracket.games[32+i].winner) {
                score += Self::calculate_win_score(1, self.games[32+i].winner.seed, config);
            }
        }
        // Round 3
        for i in 0..8{
            if Arc::ptr_eq(&self.games[48+i].winner, &referencebracket.games[48+i].winner) {
                score += Self::calculate_win_score(2, self.games[48+i].winner.seed, config);
            }
        }
        // Round 4
        for i in 0..4{
            if Arc::ptr_eq(&self.games[56+i].winner, &referencebracket.games[56+i].winner) {
                score += Self::calculate_win_score(3, self.games[56+i].winner.seed, config);
            }
        }
        // Round 5
        for i in 0..2{
            if Arc::ptr_eq(&self.games[60+i].winner, &referencebracket.games[60+i].winner) {
                score += Self::calculate_win_score(4, self.games[60+i].winner.seed, config);
            }
        }
        // Round 6
        if Arc::ptr_eq(&self.games[62].winner, &referencebracket.games[62].winner) {
            score += Self::calculate_win_score(5, self.games[62].winner.seed, config);
        }
        score
    }

    /// Fast scoring using pre-computed lookup table
    /// This is the hot path - called ~300 million times per optimization run
    #[inline]
    pub fn score_fast(&self, referencebracket: &Bracket, table: &ScoreTable) -> f64 {
        let mut score: f64 = 0.0;

        // Round 1 (games 0-31)
        for i in 0..32 {
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += table.get(0, self.games[i].winner.seed);
            }
        }
        // Round 2 (games 32-47)
        for i in 32..48 {
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += table.get(1, self.games[i].winner.seed);
            }
        }
        // Round 3 (games 48-55)
        for i in 48..56 {
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += table.get(2, self.games[i].winner.seed);
            }
        }
        // Round 4 (games 56-59)
        for i in 56..60 {
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += table.get(3, self.games[i].winner.seed);
            }
        }
        // Round 5 (games 60-61)
        for i in 60..62 {
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += table.get(4, self.games[i].winner.seed);
            }
        }
        // Round 6 (game 62)
        if Arc::ptr_eq(&self.games[62].winner, &referencebracket.games[62].winner) {
            score += table.get(5, self.games[62].winner.seed);
        }

        score
    }

    /// Create a bracket from a binary (hilo) representation.
    pub fn new_from_binary(tournamentinfo: &TournamentInfo, binary_slice: &[bool], config: Option<&ScoringConfig>) -> Bracket{
        assert!(binary_slice.len() == 63, "Binary slice must be 63 elements long");
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);

        let mut games: Vec<Game> = Vec::with_capacity(63);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;

        // Use an index to track position in binary slice
        let mut idx = 0;

        let region_names = ["East", "West", "South", "Midwest"];

        // Matchup indices for direct lookup (avoids search)
        const R2_INDICES: [(usize, usize); 4] = [(0, 7), (4, 3), (5, 2), (6, 1)];
        const R3_INDICES: [(usize, usize); 2] = [(0, 1), (2, 3)];
        const R4_INDICES: [(usize, usize); 1] = [(0, 1)];

        // Round 1
        for region in &region_names {
            for matchup in tournamentinfo.round1 {
                let team1 = tournamentinfo.get_team(region, matchup[0]);
                let team2 = tournamentinfo.get_team(region, matchup[1]);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                let win_score = Self::calculate_win_score(0, game.winner.seed, config);

                idx += 1;
                prob *= game.winnerprob;
                score += win_score;
                expected_value += win_score * game.winnerprob;
                games.push(game);
            }
        }

        // Round 2
        let r1_start = 0;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 8;
            for &(idx1, idx2) in &R2_INDICES {
                let team1 = Arc::clone(&games[r1_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r1_start + offset + idx2].winner);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                let win_score = Self::calculate_win_score(1, game.winner.seed, config);

                idx += 1;
                prob *= game.winnerprob;
                score += win_score;
                expected_value += win_score * game.winnerprob;
                games.push(game);
            }
        }

        // Round 3
        let r2_start = 32;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 4;
            for &(idx1, idx2) in &R3_INDICES {
                let team1 = Arc::clone(&games[r2_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r2_start + offset + idx2].winner);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                let win_score = Self::calculate_win_score(2, game.winner.seed, config);

                idx += 1;
                prob *= game.winnerprob;
                score += win_score;
                expected_value += win_score * game.winnerprob;
                games.push(game);
            }
        }

        // Round 4
        let r3_start = 48;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 2;
            for &(idx1, idx2) in &R4_INDICES {
                let team1 = Arc::clone(&games[r3_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r3_start + offset + idx2].winner);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                let win_score = Self::calculate_win_score(3, game.winner.seed, config);

                idx += 1;
                prob *= game.winnerprob;
                score += win_score;
                expected_value += win_score * game.winnerprob;
                games.push(game);
            }
        }

        // Final Four
        let r4_start = 56;
        let south_winner = Arc::clone(&games[r4_start + 2].winner);
        let midwest_winner = Arc::clone(&games[r4_start + 3].winner);

        let game = Game::new_from_binary(&south_winner, &midwest_winner, binary_slice[idx]);
        let win_score = Self::calculate_win_score(4, game.winner.seed, config);

        idx += 1;
        prob *= game.winnerprob;
        score += win_score;
        expected_value += win_score * game.winnerprob;
        games.push(game);

        // Final Four: East vs West
        let east_winner = Arc::clone(&games[r4_start + 0].winner);
        let west_winner = Arc::clone(&games[r4_start + 1].winner);

        let game = Game::new_from_binary(&east_winner, &west_winner, binary_slice[idx]);
        // Same round index 4
        let win_score = Self::calculate_win_score(4, game.winner.seed, config);

        idx += 1;
        prob *= game.winnerprob;
        score += win_score;
        expected_value += win_score * game.winnerprob;
        games.push(game);

        // Championship game
        let r5_start = 60;
        let team1 = Arc::clone(&games[r5_start + 0].winner);
        let team2 = Arc::clone(&games[r5_start + 1].winner);
        let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
        let win_score = Self::calculate_win_score(5, game.winner.seed, config);

        prob *= game.winnerprob;
        score += win_score;
        expected_value += win_score * game.winnerprob;

        let tournament_winner = Arc::clone(&game.winner);
        games.push(game);

        Bracket{
            games,
            winner: tournament_winner,
            prob,
            score,
            sim_score: 0.0,
            expected_value,
            binary: binary_slice.to_vec(),
        }
    }
    /// Create a mutated copy of this bracket.
    /// Uses the binary representation to efficiently create variations.
    pub fn mutate(&self, tournamentinfo: &TournamentInfo, mutation_rate: f64, config: Option<&ScoringConfig>) -> Bracket {
        let mut new_binary: Vec<bool> = self.binary.clone();
        let mut rng = rand::thread_rng();
        for bit in new_binary.iter_mut() {
            let rand: f64 = rng.gen();
            if rand < mutation_rate {
                *bit = !*bit;
            }
        }
        // Pass slice instead of owned Vec
        Bracket::new_from_binary(tournamentinfo, &new_binary, config)
    }

    pub fn create_n_children(&mut self, tournamentinfo: &TournamentInfo, n: usize, mutation_rate: f64, config: Option<&ScoringConfig>) -> Vec<Bracket>{
        let children: Vec<Bracket> = (0..n).into_iter().map(|_| self.mutate(tournamentinfo, mutation_rate, config)).collect();
        children
    }

    //not sure if this is helpful or not given score(Bracket) uses the same logic with the more relevant points
    pub fn hamming_distance(&self, other: &Bracket) -> usize{
        let mut distance = 0;
        for i in 0..self.binary.len(){
            if self.binary[i] != other.binary[i]{
                distance += 1;
            }
        }
        distance
    }

    /// Weighted distance metric that values later rounds more heavily
    /// Returns a struct with total distance, similarity percentage, and per-round breakdown
    pub fn weighted_distance(&self, other: &Bracket, config: Option<&ScoringConfig>) -> BracketDistance {
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);

        // Use configured round scores as weights
        let weights = config.round_scores;

        let mut round_distances: [f64; 6] = [0.0; 6];
        let mut round_matches: [usize; 6] = [0; 6];
        let mut round_totals: [usize; 6] = [32, 16, 8, 4, 2, 1];

        // Round 1: 32 games
        for i in 0..32 {
            if self.games[i].winner == other.games[i].winner {
                round_matches[0] += 1;
            } else {
                let seed_weight = (self.games[i].winner.seed + other.games[i].winner.seed) as f64 / 2.0;
                round_distances[0] += weights[0] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 2: 16 games
        for i in 0..16 {
            if self.games[32+i].winner == other.games[32+i].winner {
                round_matches[1] += 1;
            } else {
                let seed_weight = (self.games[32+i].winner.seed + other.games[32+i].winner.seed) as f64 / 2.0;
                round_distances[1] += weights[1] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 3 (Sweet 16): 8 games
        for i in 0..8 {
            if self.games[48+i].winner == other.games[48+i].winner {
                round_matches[2] += 1;
            } else {
                let seed_weight = (self.games[48+i].winner.seed + other.games[48+i].winner.seed) as f64 / 2.0;
                round_distances[2] += weights[2] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 4 (Elite 8): 4 games
        for i in 0..4 {
            if self.games[56+i].winner == other.games[56+i].winner {
                round_matches[3] += 1;
            } else {
                let seed_weight = (self.games[56+i].winner.seed + other.games[56+i].winner.seed) as f64 / 2.0;
                round_distances[3] += weights[3] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 5 (Final Four): 2 games
        for i in 0..2 {
            if self.games[60+i].winner == other.games[60+i].winner {
                round_matches[4] += 1;
            } else {
                let seed_weight = (self.games[60+i].winner.seed + other.games[60+i].winner.seed) as f64 / 2.0;
                round_distances[4] += weights[4] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 6 (Championship): 1 game
        if self.games[62].winner == other.games[62].winner {
            round_matches[5] += 1;
        } else {
            let seed_weight = (self.games[62].winner.seed + other.games[62].winner.seed) as f64 / 2.0;
            round_distances[5] += weights[5] * (1.0 + seed_weight / 16.0);
        }

        // Calculate total weighted distance
        let total_distance: f64 = round_distances.iter().sum();

        // Calculate maximum possible distance (using updated weights)
        let max_distance: f64 =
            32.0 * weights[0] * 2.0 +
            16.0 * weights[1] * 2.0 +
            8.0 * weights[2] * 2.0 +
            4.0 * weights[3] * 2.0 +
            2.0 * weights[4] * 2.0 +
            1.0 * weights[5] * 2.0;

        // Similarity as percentage
        let similarity = 1.0 - (total_distance / max_distance);

        BracketDistance {
            total_distance,
            similarity,
            round_distances,
            round_matches,
            round_totals,
            champion_match: self.winner == other.winner,
        }
    }

    //This will iterate through each round and print out the winner's names
    pub fn pretty_print(&self){
        println!("Round 1");
        for game in &self.games[0..32]{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Round 2");
        for game in &self.games[32..48]{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Sweet 16");
        for game in &self.games[48..56]{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Elite 8");
        for game in &self.games[56..60]{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Final Four");
        for game in &self.games[60..62]{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Championship");
        for game in &self.games[62..63]{
            println!("{} {} wins!", game.winner.seed, game.winner.name);
        }
        println!("Expected Value: {}", self.expected_value);
        println!("Maximum Score: {}", self.score);
        println!();
    }
}
pub fn random63bool() -> Vec<bool>{
    let binary: Vec<bool> = (0..63).into_iter().map(|_| {
        let mut rng = rand::thread_rng();
        let rand: f64 = rng.gen();
        if rand < 0.5{
            false
        }
        else{
            true
        }
    }).collect();
    binary
}
