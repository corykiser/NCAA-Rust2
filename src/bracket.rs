//This contains all of the simulations using ELO (modified per 538) calculations
//The Game struct is used to simulate a game between two teams and store the results.
//The Bracket struct is used to simulate a whole tournament of 63 games and store the results.
//Both structs also provide a way to create a game from binary data or to extract a binary representation of each.
// the Bracket struct also provides a way to score a bracket against a reference bracket.

use serde::{Serialize}; // Removed Deserialize as it's not used here
use rand::Rng;
use crate::ingest::{Team, TournamentInfo};
use rayon::prelude::*;

#[derive(Serialize, Debug, Clone)]
pub struct Game<'a> {
    pub team1: &'a Team,
    pub team2: &'a Team,
    pub team1prob: f64,
    pub team2prob: f64,
    pub winnerprob: f64,
    pub winner: &'a Team,
    pub hilo: bool, //did the lower seed (or the region first in alphabetical order win) win?
}

impl<'a> Game<'a> {
    pub fn new(team1: &'a Team, team2: &'a Team) -> Game<'a> {
        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let mut rng = rand::thread_rng();
        let rand_num: f64 = rng.gen(); 

        let winner: &'a Team = if rand_num < team1prob {
            team1
        } else {
            team2
        };

        let winnerprob = if rand_num < team1prob {
            team1prob
        } else {
            team2prob
        };

        let hilo: bool = if team1.region == team2.region {
            team1.seed < team2.seed
        } else {
            team1.region < team2.region
        };
        Game {
            team1,
            team2,
            team1prob,
            team2prob,
            winnerprob,
            winner,
            hilo,
        }
    }

    pub fn new_from_binary(team1: &'a Team, team2: &'a Team, hilo: bool) -> Game<'a> {
        let low_seed_team: &'a Team = if team1.seed < team2.seed { team1 } else { team2 };
        let high_seed_team: &'a Team = if team1.seed < team2.seed { team2 } else { team1 };
        let low_alpha_team: &'a Team = if team1.region < team2.region { team1 } else { team2 };
        let high_alpha_team: &'a Team = if team1.region < team2.region { team2 } else { team1 };

        let winner: &'a Team = match hilo {
            true if team1.region == team2.region => low_seed_team,
            false if team1.region == team2.region => high_seed_team,
            true if team1.region != team2.region => low_alpha_team,
            false if team1.region != team2.region => high_alpha_team,
            _ => panic!("Something went wrong in the hilo logic for new_from_binary"),
        };

        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let winnerprob = if winner == team1 { team1prob } else { team2prob };

        Game {
            team1,
            team2,
            team1prob,
            team2prob,
            winnerprob,
            winner,
            hilo,
        }
    }

    #[allow(dead_code)] // It's used in main.rs, but cargo check in sandbox might not see it
    pub fn print(&self) {
        println!("{:?}", self);
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct Bracket<'a> {
    pub round1: Vec<Game<'a>>, 
    pub round2: Vec<Game<'a>>, 
    pub round3: Vec<Game<'a>>, 
    pub round4: Vec<Game<'a>>, 
    pub round5: Vec<Game<'a>>, 
    pub round6: Vec<Game<'a>>, 
    pub winner: &'a Team, 
    pub prob: f64,  
    pub score: f64, 
    pub sim_score: f64, 
    pub expected_value: f64, 
    pub binary: Vec<bool>,
}

impl<'a> PartialEq for Bracket<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.binary == other.binary
    }
}

impl<'a> Bracket<'a> {
    pub fn new(tournamentinfo: &'a TournamentInfo) -> Bracket<'a> {
        let mut games1: Vec<Game<'a>> = Vec::with_capacity(32);
        let mut games2: Vec<Game<'a>> = Vec::with_capacity(16);
        let mut games3: Vec<Game<'a>> = Vec::with_capacity(8);
        let mut games4: Vec<Game<'a>> = Vec::with_capacity(4);
        let mut games5: Vec<Game<'a>> = Vec::with_capacity(2);
        let mut games6: Vec<Game<'a>> = Vec::with_capacity(1);

        let mut games1winners: Vec<&'a Team> = Vec::with_capacity(32);
        let mut games2winners: Vec<&'a Team> = Vec::with_capacity(16);
        let mut games3winners: Vec<&'a Team> = Vec::with_capacity(8);
        let mut games4winners: Vec<&'a Team> = Vec::with_capacity(4); 
        let mut games5winners: Vec<&'a Team> = Vec::with_capacity(2); 
        let mut games6winners: Vec<&'a Team> = Vec::with_capacity(1); 

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;
        let mut binary: Vec<bool> = Vec::with_capacity(63);
        let region_names: Vec<&str> = vec!["East", "West", "South", "Midwest"];
        
        for &region_name in &region_names {
            for r1_matchup_seeds in tournamentinfo.round1 { 
                let team1 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[0] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[0]));
                let team2 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[1] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[1]));
                let game = Game::new(team1, team2);
                games1.push(game.clone());
                games1winners.push(game.winner);
                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (1.0 + game.winner.seed as f64);
                binary.push(game.hilo);
            }
        }
        assert!(games1winners.len() == 32);

        let mut r1_winner_idx_offset = 0;
        for _ in 0..region_names.len() { 
            for _ in 0..4 { 
                let team1 = games1winners[r1_winner_idx_offset];
                let team2 = games1winners[r1_winner_idx_offset + 1];
                let game = Game::new(team1, team2);
                games2.push(game.clone());
                games2winners.push(game.winner);
                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (2.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                r1_winner_idx_offset += 2; 
            }
        }
        assert!(games2winners.len() == 16);
        
        let mut r2_winner_idx_offset = 0; 
        for _ in 0..region_names.len() { 
            for _ in 0..2 { 
                let team1 = games2winners[r2_winner_idx_offset];
                let team2 = games2winners[r2_winner_idx_offset + 1];
                let game = Game::new(team1, team2);
                games3.push(game.clone());
                games3winners.push(game.winner);
                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (4.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                r2_winner_idx_offset += 2;
            }
        }
        assert!(games3winners.len() == 8);

        let mut r3_winner_idx_offset = 0; 
        for _ in 0..region_names.len() { 
            let team1 = games3winners[r3_winner_idx_offset];
            let team2 = games3winners[r3_winner_idx_offset + 1];
            let game = Game::new(team1, team2);
            games4.push(game.clone());
            games4winners.push(game.winner); 
            prob *= game.winnerprob;
            score += 8.0 * game.winner.seed as f64;
            expected_value += game.winnerprob * (8.0 * game.winner.seed as f64);
            binary.push(game.hilo);
            r3_winner_idx_offset += 2;
        }
        assert!(games4winners.len() == 4); // E, W, S, M champions

        let team1_ew = games4winners[0]; 
        let team2_ew = games4winners[1]; 
        let game_ew = Game::new(team1_ew, team2_ew);
        games5.push(game_ew.clone());
        games5winners.push(game_ew.winner);
        prob *= game_ew.winnerprob;
        score += 16.0 * game_ew.winner.seed as f64;
        expected_value += game_ew.winnerprob * (16.0 * game_ew.winner.seed as f64);
        binary.push(game_ew.hilo);

        let team1_sm = games4winners[2]; 
        let team2_sm = games4winners[3]; 
        let game_sm = Game::new(team1_sm, team2_sm);
        games5.push(game_sm.clone());
        games5winners.push(game_sm.winner);
        prob *= game_sm.winnerprob;
        score += 16.0 * game_sm.winner.seed as f64;
        expected_value += game_sm.winnerprob * (16.0 * game_sm.winner.seed as f64);
        binary.push(game_sm.hilo);
        assert!(games5winners.len() == 2); 
        
        let game_champ = Game::new(games5winners[0], games5winners[1]);
        games6.push(game_champ.clone());
        games6winners.push(game_champ.winner);
        prob *= game_champ.winnerprob;
        score += 32.0 * game_champ.winner.seed as f64;
        expected_value += game_champ.winnerprob * (32.0 * game_champ.winner.seed as f64);
        binary.push(game_champ.hilo);

        assert!(games6winners.len() == 1);
        assert!(binary.len() == 63);

        Bracket{
            round1: games1, round2: games2, round3: games3, round4: games4, round5: games5, round6: games6,
            winner: games6winners[0], prob, score, sim_score: 0.0, expected_value, binary,
        }
    }

    pub fn score(&self, referencebracket: &Bracket<'a>) -> f64{
        let mut current_score: f64 = 0.0;
        for i in 0..32{ if self.round1[i].winner == referencebracket.round1[i].winner{ current_score += 1.0 + self.round1[i].winner.seed as f64; } }
        for i in 0..16{ if self.round2[i].winner == referencebracket.round2[i].winner{ current_score += 2.0 + self.round2[i].winner.seed as f64; } }
        for i in 0..8{  if self.round3[i].winner == referencebracket.round3[i].winner{ current_score += 4.0 + self.round3[i].winner.seed as f64; } }
        for i in 0..4{  if self.round4[i].winner == referencebracket.round4[i].winner{ current_score += 8.0 * self.round4[i].winner.seed as f64; } }
        for i in 0..2{  if self.round5[i].winner == referencebracket.round5[i].winner{ current_score += 16.0 * self.round5[i].winner.seed as f64;} }
        if self.round6[0].winner == referencebracket.round6[0].winner{ current_score += 32.0 * self.round6[0].winner.seed as f64; }
        current_score
    }

    pub fn new_from_binary(tournamentinfo: &'a TournamentInfo, binary_string: Vec<bool>) -> Bracket<'a>{
        let mut games1: Vec<Game<'a>> = Vec::with_capacity(32);
        let mut games2: Vec<Game<'a>> = Vec::with_capacity(16);
        let mut games3: Vec<Game<'a>> = Vec::with_capacity(8);
        let mut games4: Vec<Game<'a>> = Vec::with_capacity(4);
        let mut games5: Vec<Game<'a>> = Vec::with_capacity(2);
        let mut games6: Vec<Game<'a>> = Vec::with_capacity(1);

        let mut games1winners: Vec<&'a Team> = Vec::with_capacity(32);
        let mut games2winners: Vec<&'a Team> = Vec::with_capacity(16);
        let mut games3winners: Vec<&'a Team> = Vec::with_capacity(8);
        let mut games4winners: Vec<&'a Team> = Vec::with_capacity(4);
        let mut games5winners: Vec<&'a Team> = Vec::with_capacity(2);
        let mut games6winners: Vec<&'a Team> = Vec::with_capacity(1);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;

        assert!(binary_string.len() == 63, "Binary string must be 63 characters long");
        let mut hilo_iterator = binary_string.iter();
        let region_names: Vec<&str> = vec!["East", "West", "South", "Midwest"]; 
        
        for &region_name in &region_names { 
            for r1_matchup_seeds in tournamentinfo.round1 {
                let team1 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[0] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[0]));
                let team2 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[1] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[1]));
                let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R1"));
                games1.push(game.clone());
                games1winners.push(game.winner);
                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += (1.0 + game.winner.seed as f64) * game.winnerprob;
            }
        }
        assert!(games1winners.len() == 32);

        let mut r1_winner_idx_offset = 0;
        for _ in &region_names {
            for _ in 0..4 {
                let team1 = games1winners[r1_winner_idx_offset];
                let team2 = games1winners[r1_winner_idx_offset + 1];
                let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R2"));
                games2.push(game.clone());
                games2winners.push(game.winner);
                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += (2.0 + game.winner.seed as f64) * game.winnerprob;
                r1_winner_idx_offset += 2;
            }
        }
        assert!(games2winners.len() == 16);
        
        let mut r2_winner_idx_offset = 0;
        for _ in &region_names {
            for _ in 0..2 {
                let team1 = games2winners[r2_winner_idx_offset];
                let team2 = games2winners[r2_winner_idx_offset + 1];
                let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R3"));
                games3.push(game.clone());
                games3winners.push(game.winner);
                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += (4.0 + game.winner.seed as f64) * game.winnerprob;
                r2_winner_idx_offset += 2;
            }
        }
        assert!(games3winners.len() == 8);

        let mut r3_winner_idx_offset = 0;
        for _ in &region_names {
            let team1 = games3winners[r3_winner_idx_offset];
            let team2 = games3winners[r3_winner_idx_offset + 1];
            let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R4"));
            games4.push(game.clone());
            games4winners.push(game.winner);
            prob *= game.winnerprob;
            score += 8.0 * game.winner.seed as f64;
            expected_value += (8.0 * game.winner.seed as f64) * game.winnerprob;
            r3_winner_idx_offset += 2;
        }
        assert!(games4winners.len() == 4);

        let team1_ew = games4winners[0]; 
        let team2_ew = games4winners[1]; 
        let game_ew = Game::new_from_binary(team1_ew, team2_ew, *hilo_iterator.next().expect("Binary string exhausted prematurely in F4 EW"));
        games5.push(game_ew.clone());
        games5winners.push(game_ew.winner);
        prob *= game_ew.winnerprob;
        score += 16.0 * game_ew.winner.seed as f64;
        expected_value += (16.0 * game_ew.winner.seed as f64) * game_ew.winnerprob;
        
        let team1_sm = games4winners[2]; 
        let team2_sm = games4winners[3]; 
        let game_sm = Game::new_from_binary(team1_sm, team2_sm, *hilo_iterator.next().expect("Binary string exhausted prematurely in F4 SM"));
        games5.push(game_sm.clone());
        games5winners.push(game_sm.winner);
        prob *= game_sm.winnerprob;
        score += 16.0 * game_sm.winner.seed as f64;
        expected_value += (16.0 * game_sm.winner.seed as f64) * game_sm.winnerprob;
        assert!(games5winners.len() == 2);
        
        let game_champ = Game::new_from_binary(games5winners[0], games5winners[1], *hilo_iterator.next().expect("Binary string exhausted prematurely in Championship"));
        games6.push(game_champ.clone());
        games6winners.push(game_champ.winner);
        prob *= game_champ.winnerprob;
        score += 32.0 * game_champ.winner.seed as f64;
        expected_value += (32.0 * game_champ.winner.seed as f64) * game_champ.winnerprob;
        assert!(games6winners.len() == 1);
        assert!(hilo_iterator.next().is_none(), "Hilo iterator was not fully consumed.");

        Bracket{
            round1: games1, round2: games2, round3: games3, round4: games4, round5: games5, round6: games6,
            winner: games6winners[0], prob, score, sim_score: 0.0, expected_value,
            binary: binary_string, 
        }
    }

    pub fn mutate(&self,tournamentinfo: &'a TournamentInfo, mutation_rate: f64) -> Bracket<'a>{
        let mut new_binary: Vec<bool> = self.binary.clone();
        new_binary.iter_mut().for_each(|x| {
            let mut rng = rand::thread_rng();
            let rand_val: f64 = rng.gen();
            if rand_val < mutation_rate{
                *x = !*x;
            }
        });
        Bracket::new_from_binary(tournamentinfo, new_binary)
    }

    pub fn create_n_children(&mut self, tournamentinfo: &'a TournamentInfo, n: usize, mutation_rate: f64) -> Vec<Bracket<'a>>{
        (0..n).into_par_iter()
            .map(|_| self.mutate(tournamentinfo, mutation_rate))
            .collect()
    }

    #[allow(dead_code)]
    pub fn hamming_distance(&self, other: &Bracket<'a>) -> usize{
        self.binary.iter().zip(other.binary.iter()).filter(|&(a,b)| a != b).count()
    }
    
    pub fn pretty_print(&self){
        println!("Round 1");
        for game in &self.round1{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nRound 2");
        for game in &self.round2{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nSweet 16");
        for game in &self.round3{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nElite 8");
        for game in &self.round4{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nFinal Four");
        for game in &self.round5{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nChampionship");
        println!("{} {} wins!", self.round6[0].winner.seed, self.round6[0].winner.name);
        println!("Expected Value: {}", self.expected_value);
        println!("Maximum Score: {}", self.score);
        println!();
    }
}

#[allow(dead_code)]
pub fn random63bool() -> Vec<bool>{
    let mut rng = rand::thread_rng();
    (0..63).map(|_| rng.gen::<bool>()).collect()
}
