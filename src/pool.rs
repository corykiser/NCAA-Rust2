//This file is intended to help with the creation of a pool of brackets to be used in the program.
//The idea is that we can create a pool of bracket entries and run Monte Carlo Simulations against them to test their fitness.
//Fitness of a batch should be the highest MAXIMUM score of any one bracket in the batch.

//It seems like a bracket's expected score when compared against monte carlo simulations converges at around 500-1000 simulations, maybe even less.
//You probably don't even need to generate 500-1000 simulations each time if you are using an evolutionary algorithm.

//The standard deviation of how an individual bracket performs against monte carlo simulations is an interesting topic to explore.
// Finding the average score and standard deviation of a random bracket scored against monte carlo simulations could be useful for bayesian inference.
// You would want to compare that number against the average maximum score of a batch and the standard deviation of the maximum score of a batch.

//Hamming Distance between two brackets is a good metric to use to determine how similar two brackets are to each other.
//It could come in handy for maximizing diversity in a batch of brackets or for evolutionary algorithms.

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
    pub fn score_against_simulations(&mut self,tournamentinfo: &TournamentInfo, num_sims: i32){
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
    //This function will score each bracket in the batch against a reference bracket and then return the average score
    pub fn score_against_ref(&mut self, ref_bracket: &Bracket){
        self.batch_score = 0.0;
        for bracket in &mut self.brackets{
            self.batch_score += bracket.score(ref_bracket);
        }
        self.batch_score /= self.brackets.len() as f64;

    }

}