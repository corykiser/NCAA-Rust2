//This contains all of the simulations using ELO (modified per 538) calculations
//The Game struct is used to simulate a game between two teams and store the results.
//The Bracket struct is used to simulate a whole tournament of 63 games and store the results.
//Both structs also provide a way to create a game from binary data or to extract a binary representation of each.
// the Bracket struct also provides a way to score a bracket against a reference bracket.

use rand::Rng;
use std::sync::Arc;
use crate::ingest::{Team, RcTeam, TournamentInfo};

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
    /// Create a new random bracket using Monte Carlo simulation.
    ///
    /// Optimized to use:
    /// - O(1) team lookups via team_lookup map instead of O(n) filtering
    /// - RcTeam (Rc<Team>) for O(1) reference counting instead of cloning
    /// - Extracting winner before pushing game to avoid double cloning
    pub fn new(tournamentinfo: &TournamentInfo) -> Bracket{
        // Preallocate vector for all games
        let mut games: Vec<Game> = Vec::with_capacity(63);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;
        let mut binary: Vec<bool> = Vec::with_capacity(63);

        let region_names = ["East", "West", "South", "Midwest"];

        // Matchup indices for direct lookup (avoids search)
        // These correspond to the indices in the previous round's winners vector
        const R2_INDICES: [(usize, usize); 4] = [(0, 7), (4, 3), (5, 2), (6, 1)];
        const R3_INDICES: [(usize, usize); 2] = [(0, 1), (2, 3)];
        const R4_INDICES: [(usize, usize); 1] = [(0, 1)];

        // Round 1: Use O(1) team lookup instead of O(n) filtering
        for region in &region_names {
            for matchup in tournamentinfo.round1 {
                // O(1) lookup instead of filtering entire teams vector
                let team1 = tournamentinfo.get_team(region, matchup[0]);
                let team2 = tournamentinfo.get_team(region, matchup[1]);

                let game = Game::new(&team1, &team2);

                // Extract winner before pushing game (avoids cloning game)
                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (1.0 + game.winner.seed as f64);
                binary.push(game.hilo);
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
                let game = Game::new(&team1, &team2);

                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (2.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                games.push(game);
            }
        }

        // Round 3 (Sweet 16)
        let r2_start = 32;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 4;
            for &(idx1, idx2) in &R3_INDICES {
                let team1 = Arc::clone(&games[r2_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r2_start + offset + idx2].winner);
                let game = Game::new(&team1, &team2);

                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (4.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                games.push(game);
            }
        }

        // Round 4 (Elite 8)
        let r3_start = 48;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 2;
            for &(idx1, idx2) in &R4_INDICES {
                let team1 = Arc::clone(&games[r3_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r3_start + offset + idx2].winner);
                let game = Game::new(&team1, &team2);

                prob *= game.winnerprob;
                score += 8.0 * game.winner.seed as f64;
                expected_value += game.winnerprob * (8.0 * game.winner.seed as f64);
                binary.push(game.hilo);
                games.push(game);
            }
        }

        // Final Four: South vs Midwest
        // games from r4 order: East, West, South, Midwest
        let r4_start = 56;
        let south_winner = Arc::clone(&games[r4_start + 2].winner);
        let midwest_winner = Arc::clone(&games[r4_start + 3].winner);

        let game = Game::new(&south_winner, &midwest_winner);
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (16.0 * game.winner.seed as f64);
        binary.push(game.hilo);
        games.push(game);

        // Final Four: East vs West
        let east_winner = Arc::clone(&games[r4_start + 0].winner);
        let west_winner = Arc::clone(&games[r4_start + 1].winner);

        let game = Game::new(&east_winner, &west_winner);
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (16.0 * game.winner.seed as f64);
        binary.push(game.hilo);
        games.push(game);

        // Championship game
        // games from r5 order: S/MW, E/W
        let r5_start = 60;
        let team1 = Arc::clone(&games[r5_start + 0].winner);
        let team2 = Arc::clone(&games[r5_start + 1].winner);
        let game = Game::new(&team1, &team2);
        prob *= game.winnerprob;
        score += 32.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (32.0 * game.winner.seed as f64);
        binary.push(game.hilo);
        let tournament_winner = Arc::clone(&game.winner);
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
    pub fn score(&self, referencebracket: &Bracket) -> f64{
        let mut score: f64 = 0.0;
        // Using pointer comparison for performance (O(1) instead of string comparison)
        // Since all brackets use RcTeam from the same TournamentInfo, identical teams share the same Arc pointer

        //round 1
        for i in 0..32{
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += 1.0 + self.games[i].winner.seed as f64;
            }
        }
        //round 2
        for i in 0..16{
            if Arc::ptr_eq(&self.games[32+i].winner, &referencebracket.games[32+i].winner) {
                score += 2.0 + self.games[32+i].winner.seed as f64;
            }
        }
        //round 3
        for i in 0..8{
            if Arc::ptr_eq(&self.games[48+i].winner, &referencebracket.games[48+i].winner) {
                score += 4.0 + self.games[48+i].winner.seed as f64;
            }
        }
        //round 4
        for i in 0..4{
            if Arc::ptr_eq(&self.games[56+i].winner, &referencebracket.games[56+i].winner) {
                score += 8.0 * self.games[56+i].winner.seed as f64;
            }
        }
        //round 5
        for i in 0..2{
            if Arc::ptr_eq(&self.games[60+i].winner, &referencebracket.games[60+i].winner) {
                score += 16.0 * self.games[60+i].winner.seed as f64;
            }
        }
        //round 6
        if Arc::ptr_eq(&self.games[62].winner, &referencebracket.games[62].winner) {
            score += 32.0 * self.games[62].winner.seed as f64;
        }
        score
    }

    /// Create a bracket from a binary (hilo) representation.
    ///
    /// Optimized to use:
    /// - Takes a slice instead of owned Vec to avoid cloning
    /// - O(1) team lookups via team_lookup map instead of O(n) filtering
    /// - RcTeam (Rc<Team>) for O(1) reference counting instead of cloning
    pub fn new_from_binary(tournamentinfo: &TournamentInfo, binary_slice: &[bool]) -> Bracket{
        assert!(binary_slice.len() == 63, "Binary slice must be 63 elements long");

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

        // Round 1: Use O(1) team lookup
        for region in &region_names {
            for matchup in tournamentinfo.round1 {
                let team1 = tournamentinfo.get_team(region, matchup[0]);
                let team2 = tournamentinfo.get_team(region, matchup[1]);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                idx += 1;

                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += (1.0 + game.winner.seed as f64) * game.winnerprob;
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
                idx += 1;

                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += (2.0 + game.winner.seed as f64) * game.winnerprob;
                games.push(game);
            }
        }

        // Round 3 (Sweet 16)
        let r2_start = 32;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 4;
            for &(idx1, idx2) in &R3_INDICES {
                let team1 = Arc::clone(&games[r2_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r2_start + offset + idx2].winner);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                idx += 1;

                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += (4.0 + game.winner.seed as f64) * game.winnerprob;
                games.push(game);
            }
        }

        // Round 4 (Elite 8)
        let r3_start = 48;
        for (i, _) in region_names.iter().enumerate() {
            let offset = i * 2;
            for &(idx1, idx2) in &R4_INDICES {
                let team1 = Arc::clone(&games[r3_start + offset + idx1].winner);
                let team2 = Arc::clone(&games[r3_start + offset + idx2].winner);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                idx += 1;

                prob *= game.winnerprob;
                score += 8.0 * game.winner.seed as f64;
                expected_value += (8.0 * game.winner.seed as f64) * game.winnerprob;
                games.push(game);
            }
        }

        // Final Four: South vs Midwest
        let r4_start = 56;
        let south_winner = Arc::clone(&games[r4_start + 2].winner);
        let midwest_winner = Arc::clone(&games[r4_start + 3].winner);

        let game = Game::new_from_binary(&south_winner, &midwest_winner, binary_slice[idx]);
        idx += 1;
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += (16.0 * game.winner.seed as f64) * game.winnerprob;
        games.push(game);

        // Final Four: East vs West
        let east_winner = Arc::clone(&games[r4_start + 0].winner);
        let west_winner = Arc::clone(&games[r4_start + 1].winner);

        let game = Game::new_from_binary(&east_winner, &west_winner, binary_slice[idx]);
        idx += 1;
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += (16.0 * game.winner.seed as f64) * game.winnerprob;
        games.push(game);

        // Championship game
        let r5_start = 60;
        let team1 = Arc::clone(&games[r5_start + 0].winner);
        let team2 = Arc::clone(&games[r5_start + 1].winner);
        let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
        prob *= game.winnerprob;
        score += 32.0 * game.winner.seed as f64;
        expected_value += (32.0 * game.winner.seed as f64) * game.winnerprob;
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
    pub fn mutate(&self, tournamentinfo: &TournamentInfo, mutation_rate: f64) -> Bracket {
        let mut new_binary: Vec<bool> = self.binary.clone();
        let mut rng = rand::thread_rng();
        for bit in new_binary.iter_mut() {
            let rand: f64 = rng.gen();
            if rand < mutation_rate {
                *bit = !*bit;
            }
        }
        // Pass slice instead of owned Vec
        Bracket::new_from_binary(tournamentinfo, &new_binary)
    }
    pub fn create_n_children(&mut self, tournamentinfo: &TournamentInfo, n: usize, mutation_rate: f64) -> Vec<Bracket>{
        let children: Vec<Bracket> = (0..n).into_iter().map(|_| self.mutate(tournamentinfo, mutation_rate)).collect();
        // let mut children: Vec<Bracket> = Vec::new();
        // for _ in 0..n{
        //     children.push(self.mutate(tournamentinfo, mutation_rate));
        // }

        // //add self to children
        // children.push(self.clone());
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
    pub fn weighted_distance(&self, other: &Bracket) -> BracketDistance {
        // Round weights based on scoring system
        // Round 1: base 1, Round 2: base 2, etc. up to Championship: base 32
        const ROUND_WEIGHTS: [f64; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

        let mut round_distances: [f64; 6] = [0.0; 6];
        let mut round_matches: [usize; 6] = [0; 6];
        let mut round_totals: [usize; 6] = [32, 16, 8, 4, 2, 1];

        // Round 1: 32 games (indices 0-31 in binary)
        for i in 0..32 {
            if self.games[i].winner == other.games[i].winner {
                round_matches[0] += 1;
            } else {
                // Weight by average seed of the differing winner
                let seed_weight = (self.games[i].winner.seed + other.games[i].winner.seed) as f64 / 2.0;
                round_distances[0] += ROUND_WEIGHTS[0] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 2: 16 games
        for i in 0..16 {
            if self.games[32+i].winner == other.games[32+i].winner {
                round_matches[1] += 1;
            } else {
                let seed_weight = (self.games[32+i].winner.seed + other.games[32+i].winner.seed) as f64 / 2.0;
                round_distances[1] += ROUND_WEIGHTS[1] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 3 (Sweet 16): 8 games
        for i in 0..8 {
            if self.games[48+i].winner == other.games[48+i].winner {
                round_matches[2] += 1;
            } else {
                let seed_weight = (self.games[48+i].winner.seed + other.games[48+i].winner.seed) as f64 / 2.0;
                round_distances[2] += ROUND_WEIGHTS[2] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 4 (Elite 8): 4 games
        for i in 0..4 {
            if self.games[56+i].winner == other.games[56+i].winner {
                round_matches[3] += 1;
            } else {
                let seed_weight = (self.games[56+i].winner.seed + other.games[56+i].winner.seed) as f64 / 2.0;
                round_distances[3] += ROUND_WEIGHTS[3] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 5 (Final Four): 2 games
        for i in 0..2 {
            if self.games[60+i].winner == other.games[60+i].winner {
                round_matches[4] += 1;
            } else {
                let seed_weight = (self.games[60+i].winner.seed + other.games[60+i].winner.seed) as f64 / 2.0;
                round_distances[4] += ROUND_WEIGHTS[4] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 6 (Championship): 1 game
        if self.games[62].winner == other.games[62].winner {
            round_matches[5] += 1;
        } else {
            let seed_weight = (self.games[62].winner.seed + other.games[62].winner.seed) as f64 / 2.0;
            round_distances[5] += ROUND_WEIGHTS[5] * (1.0 + seed_weight / 16.0);
        }

        // Calculate total weighted distance
        let total_distance: f64 = round_distances.iter().sum();

        // Calculate maximum possible distance (if all games differed with seed 16 teams)
        // Max per round = num_games * weight * (1 + 16/16) = num_games * weight * 2
        let max_distance: f64 =
            32.0 * ROUND_WEIGHTS[0] * 2.0 +
            16.0 * ROUND_WEIGHTS[1] * 2.0 +
            8.0 * ROUND_WEIGHTS[2] * 2.0 +
            4.0 * ROUND_WEIGHTS[3] * 2.0 +
            2.0 * ROUND_WEIGHTS[4] * 2.0 +
            1.0 * ROUND_WEIGHTS[5] * 2.0;

        // Similarity as percentage (1.0 = identical, 0.0 = maximally different)
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
