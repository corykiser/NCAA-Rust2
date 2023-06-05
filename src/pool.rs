//It seems like a bracket's expected score when compared against monte carlo simulations converges at around 500-1000 simulations, maybe even less.

use crate::bracket::{Bracket, self};
use crate::ingest::TournamentInfo;

pub struct Batch{
    pub brackets: Vec<Bracket>,
    pub batch_score: f64,
}

impl Batch{
    //This function will create a batch of brackets from MonteCarlo simulations
    pub fn new(tournamentinfo: &TournamentInfo, num_brackets: i32) -> Batch{
        let mut brackets: Vec<Bracket> = Vec::new();
        for _i in 0..num_brackets{
            brackets.push(Bracket::new(tournamentinfo));
        }
        Batch{
            brackets,
            batch_score: 0.0,
        }
    }
    // This function will score each bracket in the batch against random sims and then return the average score
    pub fn simulate_and_score(&mut self,tournamentinfo: &TournamentInfo, num_sims: i32){
        //start sims
        for i in 0..num_sims{
            let sim_bracket = Bracket::new(&tournamentinfo);
            for bracket in &mut self.brackets{
                bracket.sim_score = 0.0; //reset to zero
            }
            //begin averaging for each individual bracket score
            for bracket in &mut self.brackets{
                bracket.sim_score += bracket.score(&sim_bracket);
            }
            for bracket in &mut self.brackets{
                bracket.sim_score /= num_sims as f64;
            }
        }
        //go back and average all of the scores into the batch score
        for bracket in &mut self.brackets{
            self.batch_score += bracket.sim_score;
        }
        self.batch_score /= self.brackets.len() as f64;
    }
    pub fn score_against_ref(&mut self, ref_bracket: &Bracket){
        self.batch_score = 0.0;
        for bracket in &mut self.brackets{
            self.batch_score += bracket.score(ref_bracket);
        }
        self.batch_score /= self.brackets.len() as f64;

    }

}