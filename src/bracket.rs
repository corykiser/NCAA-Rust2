//This contains all of the simulations using ELO (modified per 538) calculations
//The Game struct is used to simulate a game between two teams and store the results.
//The Bracket struct is used to simulate a whole tournament of 63 games and store the results.
//Both structs also provide a way to create a game from binary data or to extract a binary representation of each.
// the Bracket struct also provides a way to score a bracket against a reference bracket.




//It would be nice to have a function that fills in the the rest of a bracket if you have decided to pick a certain team(s) to win.

use serde::{Serialize, Deserialize};
use rand::Rng;
use crate::ingest::{Team, TournamentInfo};
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Game {
    pub team1: Team,
    pub team2: Team,
    pub team1prob: f64,
    pub team2prob: f64,
    pub winnerprob: f64,
    pub winner: Team,
    pub hilo: bool, //did the lower seed (or the region first in alphabetical order win) win?
}


//TODO Cloning the teams is really inefficient. I should just pass out references of the teams and then clone them when I need them.
impl Game {
    //This will simulate a game between two teams
    pub fn new(team1: &Team, team2: &Team) -> Game {
        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let mut rng = rand::thread_rng();
        let rand_num: f64 = rng.gen(); // generates a float between 0 and 1

        let winner: Team = if rand_num < team1prob {
            team1.clone()
        } else {
            team2.clone()
        };

        let winnerprob = if rand_num < team1prob {
            team1prob
        } else {
            team2prob
        };

        // Check if the regions are the same. If so, did the lower seed win? If not, did the region first in alphabetical order win?
        let hilo: bool = if team1.region == team2.region {
            team1.seed < team2.seed
        } else {
            team1.region < team2.region
        };
        Game {
            team1: team1.clone(),
            team2: team2.clone(),
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
    pub fn new_from_binary(team1: &Team, team2: &Team, hilo: bool) -> Game {
        //find the lower seed team
        let low_seed_team: Team = if team1.seed < team2.seed {
            team1.clone()
        } else {
            team2.clone()
        };
        //find the higher seed team
        let high_seed_team: Team = if team1.seed < team2.seed {
            team2.clone()
        } else {
            team1.clone()
        };
        //find which region is first alphabetically
        let low_alpha_team: Team = if team1.region < team2.region {
            team1.clone()
        } else {
            team2.clone()
        };
        //find which region is second alphabetically
        let high_alpha_team: Team = if team1.region < team2.region {
            team2.clone()
        } else {
            team1.clone()
        };

        let winner: Team = match hilo {
            true if team1.region==team2.region => low_seed_team.clone(),
            false if team1.region==team2.region => high_seed_team.clone(),
            true if team1.region!=team2.region => low_alpha_team.clone(),
            false if team1.region!=team2.region => high_alpha_team.clone(),
            _ => {panic!("Something went wrong in the hilo logic")
        }
            
        };

        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let winnerprob = if winner == *team1 {
            team1prob
        } else {
            team2prob
        };

        Game {
            team1: team1.clone(),
            team2: team2.clone(),
            team1prob,
            team2prob,
            winnerprob,
            winner,
            hilo,
        }
    }
    pub fn print(&self) {
        println!("{:?}", self);
    }
}

//todo smart mutate function that only mutates the parts of the bracket that are uncertain (i.e. win probability is between 0.4 and 0.6ç)

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bracket {
    pub round1: Vec<Game>, //round of 64
    pub round2: Vec<Game>, //round of 32
    pub round3: Vec<Game>, //round of 16
    pub round4: Vec<Game>, //round of 8
    pub round5: Vec<Game>, //round of 4
    pub round6: Vec<Game>, //round of 2
    pub winner: Team, //winner of tournament
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
    pub fn new(tournamentinfo: &TournamentInfo) -> Bracket{
        //preallocating makes this faster... supposedly
        let mut games1: Vec<Game> = Vec::with_capacity(32);
        let mut games2: Vec<Game> = Vec::with_capacity(16);
        let mut games3: Vec<Game> = Vec::with_capacity(8);
        let mut games4: Vec<Game> = Vec::with_capacity(4);
        let mut games5: Vec<Game> = Vec::with_capacity(2);
        let mut games6: Vec<Game> = Vec::with_capacity(1);

        let mut games1winners: Vec<Team> = Vec::with_capacity(32);
        let mut games2winners: Vec<Team> = Vec::with_capacity(16);
        let mut games3winners: Vec<Team> = Vec::with_capacity(8);
        let mut games4winners: Vec<Team> = Vec::with_capacity(4);
        let mut games5winners: Vec<Team> = Vec::with_capacity(2);
        let mut games6winners: Vec<Team> = Vec::with_capacity(1);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;

        //TODO USE A VECTOR INSTEAD OF ARRAY?
        //array of bools to represent the bracket
        let mut binary: Vec<bool> = Vec::with_capacity(63);

        //Use the following for quick cycling through the 4 regions
        let region_names: Vec<String> = vec!["East".to_string(), "West".to_string(), "South".to_string(), "Midwest".to_string()];

        //Round 1
        for region in &region_names{
            for matchup in tournamentinfo.round1{
                let matching_teams: Vec<Team> = tournamentinfo.teams.iter().filter(|&x| x.region == *region && (x.seed == matchup[0] || x.seed == matchup[1])).cloned().collect();
                let game = Game::new(&matching_teams[0].clone(), &matching_teams[1].clone());
                games1.push(game.clone());
                games1winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (1.0 + game.winner.seed as f64);

                binary.push(game.hilo);
            }
        }
        assert!(games1winners.len() == 32);

        //Round 2
        for region in &region_names{
            for matchup in tournamentinfo.round2{
                let matching_teams: Vec<Team> = games1winners.iter().filter(|&x| x.region == *region && matchup.contains(&x.seed)).cloned().collect();
                let game = Game::new(&matching_teams[0].clone(), &matching_teams[1].clone());
                games2.push(game.clone());
                games2winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (2.0 + game.winner.seed as f64);

                binary.push(game.hilo);
            }
        }
        assert!(games2winners.len() == 16);
        
        //Round 3
        for region in &region_names{
            for matchup in tournamentinfo.round3{
                let matching_teams: Vec<Team> = games2winners.iter().filter(|&x| x.region == *region && matchup.contains(&x.seed)).cloned().collect();
                let game = Game::new(&matching_teams[0].clone(), &matching_teams[1].clone());
                games3.push(game.clone());
                games3winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (4.0 + game.winner.seed as f64);

                binary.push(game.hilo);
            }
        }
        assert!(games3winners.len() == 8);

        //Round 4
        for region in &region_names{
            for matchup in tournamentinfo.round4{
                let matching_teams: Vec<Team> = games3winners.iter().filter(|&x| x.region == *region && matchup.contains(&x.seed)).cloned().collect();
                let game = Game::new(&matching_teams[0].clone(), &matching_teams[1].clone());
                games4.push(game.clone());
                games4winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 8.0 * game.winner.seed as f64;
                expected_value += game.winnerprob * (8.0 * game.winner.seed as f64);

                binary.push(game.hilo);
            }
        }
        assert!(games4winners.len() == 4);

        //Match South with Midwest
        let matching_teams: Vec<Team> = games4winners.iter().filter(|&x| x.region == "South" || x.region == "Midwest").cloned().collect();
        let game = Game::new(&matching_teams[0].clone(), &matching_teams[1].clone());
        games5.push(game.clone());
        games5winners.push(game.clone().winner);

        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (16.0 * game.winner.seed as f64);

        binary.push(game.hilo);

        //Match East with West
        let matching_teams: Vec<Team> = games4winners.iter().filter(|&x| x.region == "East" || x.region == "West").cloned().collect();
        let game = Game::new(&matching_teams[0].clone(), &matching_teams[1].clone());
        games5.push(game.clone());
        games5winners.push(game.clone().winner);
        assert!(games5winners.len() == 2);

        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (16.0 * game.winner.seed as f64);

        binary.push(game.hilo);
        
        //championship game
        let game = Game::new(&games5winners[0].clone(), &games5winners[1].clone());
        games6.push(game.clone());
        games6winners.push(game.clone().winner);

        prob *= game.winnerprob;
        score += 32.0 * game.winner.seed as f64;
        expected_value += game.winnerprob * (32.0 * game.winner.seed as f64);

        binary.push(game.hilo);
        assert!(games6winners.len() == 1);
        assert!(binary.len() == 63);

        Bracket{
            round1: games1,
            round2: games2,
            round3: games3,
            round4: games4,
            round5: games5,
            round6: games6,
            winner: games6winners[0].clone(),
            prob,
            score,
            sim_score: 0.0,
            expected_value,
            binary,
        }
    }
    pub fn score(&self, referencebracket: &Bracket) -> f64{
        let mut score: f64 = 0.0;
        //round 1
        for i in 0..32{
            if self.round1[i].winner == referencebracket.round1[i].winner{
                score += 1.0 + self.round1[i].winner.seed as f64;
            }
        }
        //round 2
        for i in 0..16{
            if self.round2[i].winner == referencebracket.round2[i].winner{
                score += 2.0 + self.round2[i].winner.seed as f64;
            }
        }
        //round 3
        for i in 0..8{
            if self.round3[i].winner == referencebracket.round3[i].winner{
                score += 4.0 + self.round3[i].winner.seed as f64;
            }
        }
        //round 4
        for i in 0..4{
            if self.round4[i].winner == referencebracket.round4[i].winner{
                score += 8.0 * self.round4[i].winner.seed as f64;
            }
        }
        //round 5
        for i in 0..2{
            if self.round5[i].winner == referencebracket.round5[i].winner{
                score += 16.0 * self.round5[i].winner.seed as f64;
            }
        }
        //round 6
        if self.round6[0].winner == referencebracket.round6[0].winner{
            score += 32.0 * self.round6[0].winner.seed as f64;
        }
        score
    }
    pub fn new_from_binary(tournamentinfo: &TournamentInfo, mut binary_string: Vec<bool>) -> Bracket{
        let mut games1: Vec<Game> = Vec::with_capacity(32);
        let mut games2: Vec<Game> = Vec::with_capacity(16);
        let mut games3: Vec<Game> = Vec::with_capacity(8);
        let mut games4: Vec<Game> = Vec::with_capacity(4);
        let mut games5: Vec<Game> = Vec::with_capacity(2);
        let mut games6: Vec<Game> = Vec::with_capacity(1);

        let mut games1winners: Vec<Team> = Vec::with_capacity(32);
        let mut games2winners: Vec<Team> = Vec::with_capacity(16);
        let mut games3winners: Vec<Team> = Vec::with_capacity(8);
        let mut games4winners: Vec<Team> = Vec::with_capacity(4);
        let mut games5winners: Vec<Team> = Vec::with_capacity(2);
        let mut games6winners: Vec<Team> = Vec::with_capacity(1);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;

        assert!(binary_string.len() == 63, "Binary string must be 63 characters long");
        let binary: Vec<bool> = binary_string.clone(); //reuse the input string for the output binary string

        //Use the following for quick cycling through the 4 regions
        let region_names: Vec<String> = vec!["East".to_string(), "West".to_string(), "South".to_string(), "Midwest".to_string()];

        //maybe it's sloppy to not do the first four round in a loop, but I think it's more readable this way
        //Round 1
        for region in &region_names{
            for matchup in tournamentinfo.round1{
                let matching_teams: Vec<Team> = tournamentinfo.teams.iter().filter(|&x| x.region == *region && (x.seed == matchup[0] || x.seed == matchup[1])).cloned().collect();
                let game = Game::new_from_binary(&matching_teams[0].clone(), &matching_teams[1].clone(), binary_string.remove(0));
                games1.push(game.clone());
                games1winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += (1.0 + game.winner.seed as f64) * game.winnerprob;
            }
        }
        assert!(games1winners.len() == 32);

        //Round 2
        for region in &region_names{
            for matchup in tournamentinfo.round2{
                let matching_teams: Vec<Team> = games1winners.iter().filter(|&x| x.region == *region && matchup.contains(&x.seed)).cloned().collect();
                let game = Game::new_from_binary(&matching_teams[0].clone(), &matching_teams[1].clone(), binary_string.remove(0));
                games2.push(game.clone());
                games2winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += (2.0 + game.winner.seed as f64) * game.winnerprob;
            }
        }
        assert!(games2winners.len() == 16);
        
        //Round 3
        for region in &region_names{
            for matchup in tournamentinfo.round3{
                let matching_teams: Vec<Team> = games2winners.iter().filter(|&x| x.region == *region && matchup.contains(&x.seed)).cloned().collect();
                let game = Game::new_from_binary(&matching_teams[0].clone(), &matching_teams[1].clone(), binary_string.remove(0));
                games3.push(game.clone());
                games3winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += (4.0 + game.winner.seed as f64) * game.winnerprob;
            }
        }
        assert!(games3winners.len() == 8);

        //Round 4
        for region in &region_names{
            for matchup in tournamentinfo.round4{
                let matching_teams: Vec<Team> = games3winners.iter().filter(|&x| x.region == *region && matchup.contains(&x.seed)).cloned().collect();
                let game = Game::new_from_binary(&matching_teams[0].clone(), &matching_teams[1].clone(), binary_string.remove(0));
                games4.push(game.clone());
                games4winners.push(game.clone().winner);

                prob *= game.winnerprob;
                score += 8.0 * game.winner.seed as f64;
                expected_value += (8.0 * game.winner.seed as f64) * game.winnerprob;
            }
        }
        assert!(games4winners.len() == 4);

        //Match South with Midwest
        let matching_teams: Vec<Team> = games4winners.iter().filter(|&x| x.region == "South" || x.region == "Midwest").cloned().collect();
        let game = Game::new_from_binary(&matching_teams[0], &matching_teams[1], binary_string.remove(0));
        games5.push(game.clone());
        games5winners.push(game.clone().winner);

        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += (16.0 * game.winner.seed as f64) * game.winnerprob;

        //Match East with West
        let matching_teams: Vec<Team> = games4winners.iter().filter(|&x| x.region == "East" || x.region == "West").cloned().collect();
        let game = Game::new_from_binary(&matching_teams[0], &matching_teams[1], binary_string.remove(0));
        games5.push(game.clone());
        games5winners.push(game.clone().winner);
        assert!(games5winners.len() == 2);

        prob *= game.winnerprob;
        score += 16.0 * game.winner.seed as f64;
        expected_value += (16.0 * game.winner.seed as f64) * game.winnerprob;
        
        //championship game
        let game = Game::new_from_binary(&games5winners[0], &games5winners[1], binary_string.remove(0));
        games6.push(game.clone());
        games6winners.push(game.clone().winner);

        prob *= game.winnerprob;
        score += 32.0 * game.winner.seed as f64;
        expected_value += (32.0 * game.winner.seed as f64) * game.winnerprob;
        assert!(games6winners.len() == 1);

        Bracket{
            round1: games1,
            round2: games2,
            round3: games3,
            round4: games4,
            round5: games5,
            round6: games6,
            winner: games6winners[0].clone(),
            prob,
            score,
            sim_score: 0.0,
            expected_value: expected_value,
            binary: binary,
        }
    }
    pub fn mutate(&self,tournamentinfo: &TournamentInfo, mutation_rate: f64) -> Bracket{
        let mut new_binary: Vec<bool> = self.binary.clone();
        new_binary.iter_mut().for_each(|x| {
            let mut rng = rand::thread_rng();
            let rand: f64 = rng.gen();
            if rand < mutation_rate{
                *x = !*x;
            }
        });
        // for i in 0..new_binary.len(){
        //     let rand: f64 = rng.gen();
        //     if rand < mutation_rate{
        //         new_binary[i] = !new_binary[i];
        //     }
        // }
        Bracket::new_from_binary(tournamentinfo, new_binary)
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
