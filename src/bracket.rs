//This contains all of the simulations using ELO (modified per 538) calculations
//The Game struct is used to simulate a game between two teams and store the results.
//The Bracket struct is used to simulate a whole tournament of 63 games and store the results.
//Both structs also provide a way to create a game from binary data or to extract a binary representation of each.
// the Bracket struct also provides a way to score a bracket against a reference bracket.




//It would be nice to have a function that fills in the the rest of a bracket if you have decided to pick a certain team(s) to win.

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
    pub round1: Vec<Game>, //round of 64
    pub round2: Vec<Game>, //round of 32
    pub round3: Vec<Game>, //round of 16
    pub round4: Vec<Game>, //round of 8
    pub round5: Vec<Game>, //round of 4
    pub round6: Vec<Game>, //round of 2
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
        // Preallocate vectors
        let mut games1: Vec<Game> = Vec::with_capacity(32);
        let mut games2: Vec<Game> = Vec::with_capacity(16);
        let mut games3: Vec<Game> = Vec::with_capacity(8);
        let mut games4: Vec<Game> = Vec::with_capacity(4);
        let mut games5: Vec<Game> = Vec::with_capacity(2);
        let mut games6: Vec<Game> = Vec::with_capacity(1);

        // Winners are stored as RcTeam (cheap reference counting)
        let mut games1winners: Vec<RcTeam> = Vec::with_capacity(32);
        let mut games2winners: Vec<RcTeam> = Vec::with_capacity(16);
        let mut games3winners: Vec<RcTeam> = Vec::with_capacity(8);
        let mut games4winners: Vec<RcTeam> = Vec::with_capacity(4);
        let mut games5winners: Vec<RcTeam> = Vec::with_capacity(2);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;
        let mut binary: Vec<bool> = Vec::with_capacity(63);

        let region_names = ["East", "West", "South", "Midwest"];

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
                games1winners.push(Arc::clone(&game.winner));
                games1.push(game);
            }
        }
        debug_assert!(games1winners.len() == 32);

        // Helper to find winner by region and matching seeds
        #[inline]
        fn find_teams_for_matchup(winners: &[RcTeam], region: &str, matchup: &[i32]) -> (RcTeam, RcTeam) {
            let mut found: Vec<RcTeam> = Vec::with_capacity(2);
            for w in winners {
                if w.region == region && matchup.contains(&w.seed) {
                    found.push(Arc::clone(w));
                    if found.len() == 2 { break; }
                }
            }
            (found.remove(0), found.remove(0))
        }

        // Round 2
        for region in &region_names {
            for matchup in tournamentinfo.round2 {
                let (team1, team2) = find_teams_for_matchup(&games1winners, region, &matchup);
                let game = Game::new(&team1, &team2);

                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (2.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                games2winners.push(Arc::clone(&game.winner));
                games2.push(game);
            }
        }
        debug_assert!(games2winners.len() == 16);

        // Round 3 (Sweet 16)
        for region in &region_names {
            for matchup in tournamentinfo.round3 {
                let (team1, team2) = find_teams_for_matchup(&games2winners, region, &matchup);
                let game = Game::new(&team1, &team2);

                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (4.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                games3winners.push(Arc::clone(&game.winner));
                games3.push(game);
            }
        }
        debug_assert!(games3winners.len() == 8);

        // Round 4 (Elite 8)
        for region in &region_names {
            for matchup in tournamentinfo.round4 {
                let (team1, team2) = find_teams_for_matchup(&games3winners, region, &matchup);
                let game = Game::new(&team1, &team2);

                prob *= game.winnerprob;
                score += 8.0 * game.winner.seed as f64;
                expected_value += game.winnerprob * (8.0 * game.winner.seed as f64);
                binary.push(game.hilo);
                games4winners.push(Arc::clone(&game.winner));
                games4.push(game);
            }
        }
        debug_assert!(games4winners.len() == 4);

        // Final Four: South vs Midwest
        let (south_winner, midwest_winner) = {
            let mut south = None;
            let mut midwest = None;
            for w in &games4winners {
                match w.region.as_str() {
                    "South" => south = Some(Arc::clone(w)),
                    "Midwest" => midwest = Some(Arc::clone(w)),
                    _ => {}
                }
            }
            (south.unwrap(), midwest.unwrap())
        };
        let game = Game::new(&south_winner, &midwest_winner);
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (16.0 * game.winner.seed as f64);
        binary.push(game.hilo);
        games5winners.push(Arc::clone(&game.winner));
        games5.push(game);

        // Final Four: East vs West
        let (east_winner, west_winner) = {
            let mut east = None;
            let mut west = None;
            for w in &games4winners {
                match w.region.as_str() {
                    "East" => east = Some(Arc::clone(w)),
                    "West" => west = Some(Arc::clone(w)),
                    _ => {}
                }
            }
            (east.unwrap(), west.unwrap())
        };
        let game = Game::new(&east_winner, &west_winner);
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (16.0 * game.winner.seed as f64);
        binary.push(game.hilo);
        games5winners.push(Arc::clone(&game.winner));
        games5.push(game);
        debug_assert!(games5winners.len() == 2);

        // Championship game
        let game = Game::new(&games5winners[0], &games5winners[1]);
        prob *= game.winnerprob;
        score += 32.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (32.0 * game.winner.seed as f64);
        binary.push(game.hilo);
        let tournament_winner = Arc::clone(&game.winner);
        games6.push(game);
        debug_assert!(binary.len() == 63);

        Bracket{
            round1: games1,
            round2: games2,
            round3: games3,
            round4: games4,
            round5: games5,
            round6: games6,
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
        // Using name comparison (Team's PartialEq compares by name)
        //round 1
        for i in 0..32{
            if self.round1[i].winner.name == referencebracket.round1[i].winner.name{
                score += 1.0 + self.round1[i].winner.seed as f64;
            }
        }
        //round 2
        for i in 0..16{
            if self.round2[i].winner.name == referencebracket.round2[i].winner.name{
                score += 2.0 + self.round2[i].winner.seed as f64;
            }
        }
        //round 3
        for i in 0..8{
            if self.round3[i].winner.name == referencebracket.round3[i].winner.name{
                score += 4.0 + self.round3[i].winner.seed as f64;
            }
        }
        //round 4
        for i in 0..4{
            if self.round4[i].winner.name == referencebracket.round4[i].winner.name{
                score += 8.0 * self.round4[i].winner.seed as f64;
            }
        }
        //round 5
        for i in 0..2{
            if self.round5[i].winner.name == referencebracket.round5[i].winner.name{
                score += 16.0 * self.round5[i].winner.seed as f64;
            }
        }
        //round 6
        if self.round6[0].winner.name == referencebracket.round6[0].winner.name{
            score += 32.0 * self.round6[0].winner.seed as f64;
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

        let mut games1: Vec<Game> = Vec::with_capacity(32);
        let mut games2: Vec<Game> = Vec::with_capacity(16);
        let mut games3: Vec<Game> = Vec::with_capacity(8);
        let mut games4: Vec<Game> = Vec::with_capacity(4);
        let mut games5: Vec<Game> = Vec::with_capacity(2);
        let mut games6: Vec<Game> = Vec::with_capacity(1);

        let mut games1winners: Vec<RcTeam> = Vec::with_capacity(32);
        let mut games2winners: Vec<RcTeam> = Vec::with_capacity(16);
        let mut games3winners: Vec<RcTeam> = Vec::with_capacity(8);
        let mut games4winners: Vec<RcTeam> = Vec::with_capacity(4);
        let mut games5winners: Vec<RcTeam> = Vec::with_capacity(2);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;

        // Use an index to track position in binary slice
        let mut idx = 0;

        let region_names = ["East", "West", "South", "Midwest"];

        // Helper to find teams for matchup (for rounds after round 1)
        #[inline]
        fn find_teams_for_matchup(winners: &[RcTeam], region: &str, matchup: &[i32]) -> (RcTeam, RcTeam) {
            let mut found: Vec<RcTeam> = Vec::with_capacity(2);
            for w in winners {
                if w.region == region && matchup.contains(&w.seed) {
                    found.push(Arc::clone(w));
                    if found.len() == 2 { break; }
                }
            }
            (found.remove(0), found.remove(0))
        }

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
                games1winners.push(Arc::clone(&game.winner));
                games1.push(game);
            }
        }
        debug_assert!(games1winners.len() == 32);

        // Round 2
        for region in &region_names {
            for matchup in tournamentinfo.round2 {
                let (team1, team2) = find_teams_for_matchup(&games1winners, region, &matchup);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                idx += 1;

                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += (2.0 + game.winner.seed as f64) * game.winnerprob;
                games2winners.push(Arc::clone(&game.winner));
                games2.push(game);
            }
        }
        debug_assert!(games2winners.len() == 16);

        // Round 3 (Sweet 16)
        for region in &region_names {
            for matchup in tournamentinfo.round3 {
                let (team1, team2) = find_teams_for_matchup(&games2winners, region, &matchup);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                idx += 1;

                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += (4.0 + game.winner.seed as f64) * game.winnerprob;
                games3winners.push(Arc::clone(&game.winner));
                games3.push(game);
            }
        }
        debug_assert!(games3winners.len() == 8);

        // Round 4 (Elite 8)
        for region in &region_names {
            for matchup in tournamentinfo.round4 {
                let (team1, team2) = find_teams_for_matchup(&games3winners, region, &matchup);
                let game = Game::new_from_binary(&team1, &team2, binary_slice[idx]);
                idx += 1;

                prob *= game.winnerprob;
                score += 8.0 * game.winner.seed as f64;
                expected_value += (8.0 * game.winner.seed as f64) * game.winnerprob;
                games4winners.push(Arc::clone(&game.winner));
                games4.push(game);
            }
        }
        debug_assert!(games4winners.len() == 4);

        // Final Four: South vs Midwest
        let (south_winner, midwest_winner) = {
            let mut south = None;
            let mut midwest = None;
            for w in &games4winners {
                match w.region.as_str() {
                    "South" => south = Some(Arc::clone(w)),
                    "Midwest" => midwest = Some(Arc::clone(w)),
                    _ => {}
                }
            }
            (south.unwrap(), midwest.unwrap())
        };
        let game = Game::new_from_binary(&south_winner, &midwest_winner, binary_slice[idx]);
        idx += 1;
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += (16.0 * game.winner.seed as f64) * game.winnerprob;
        games5winners.push(Arc::clone(&game.winner));
        games5.push(game);

        // Final Four: East vs West
        let (east_winner, west_winner) = {
            let mut east = None;
            let mut west = None;
            for w in &games4winners {
                match w.region.as_str() {
                    "East" => east = Some(Arc::clone(w)),
                    "West" => west = Some(Arc::clone(w)),
                    _ => {}
                }
            }
            (east.unwrap(), west.unwrap())
        };
        let game = Game::new_from_binary(&east_winner, &west_winner, binary_slice[idx]);
        idx += 1;
        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += (16.0 * game.winner.seed as f64) * game.winnerprob;
        games5winners.push(Arc::clone(&game.winner));
        games5.push(game);
        debug_assert!(games5winners.len() == 2);

        // Championship game
        let game = Game::new_from_binary(&games5winners[0], &games5winners[1], binary_slice[idx]);
        prob *= game.winnerprob;
        score += 32.0 * game.winner.seed as f64;
        expected_value += (32.0 * game.winner.seed as f64) * game.winnerprob;
        let tournament_winner = Arc::clone(&game.winner);
        games6.push(game);

        Bracket{
            round1: games1,
            round2: games2,
            round3: games3,
            round4: games4,
            round5: games5,
            round6: games6,
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
            if self.round1[i].winner == other.round1[i].winner {
                round_matches[0] += 1;
            } else {
                // Weight by average seed of the differing winner
                let seed_weight = (self.round1[i].winner.seed + other.round1[i].winner.seed) as f64 / 2.0;
                round_distances[0] += ROUND_WEIGHTS[0] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 2: 16 games
        for i in 0..16 {
            if self.round2[i].winner == other.round2[i].winner {
                round_matches[1] += 1;
            } else {
                let seed_weight = (self.round2[i].winner.seed + other.round2[i].winner.seed) as f64 / 2.0;
                round_distances[1] += ROUND_WEIGHTS[1] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 3 (Sweet 16): 8 games
        for i in 0..8 {
            if self.round3[i].winner == other.round3[i].winner {
                round_matches[2] += 1;
            } else {
                let seed_weight = (self.round3[i].winner.seed + other.round3[i].winner.seed) as f64 / 2.0;
                round_distances[2] += ROUND_WEIGHTS[2] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 4 (Elite 8): 4 games
        for i in 0..4 {
            if self.round4[i].winner == other.round4[i].winner {
                round_matches[3] += 1;
            } else {
                let seed_weight = (self.round4[i].winner.seed + other.round4[i].winner.seed) as f64 / 2.0;
                round_distances[3] += ROUND_WEIGHTS[3] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 5 (Final Four): 2 games
        for i in 0..2 {
            if self.round5[i].winner == other.round5[i].winner {
                round_matches[4] += 1;
            } else {
                let seed_weight = (self.round5[i].winner.seed + other.round5[i].winner.seed) as f64 / 2.0;
                round_distances[4] += ROUND_WEIGHTS[4] * (1.0 + seed_weight / 16.0);
            }
        }

        // Round 6 (Championship): 1 game
        if self.round6[0].winner == other.round6[0].winner {
            round_matches[5] += 1;
        } else {
            let seed_weight = (self.round6[0].winner.seed + other.round6[0].winner.seed) as f64 / 2.0;
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
        for game in &self.round1{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Round 2");
        for game in &self.round2{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Sweet 16");
        for game in &self.round3{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Elite 8");
        for game in &self.round4{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Final Four");
        for game in &self.round5{
            println!("{} {}", game.winner.seed, game.winner.name);
        }
        println!();
        println!("Championship");
        for game in &self.round6{
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
