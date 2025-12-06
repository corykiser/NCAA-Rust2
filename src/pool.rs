use crate::ingest::TournamentInfo;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use num_cpus;
use serde_json;

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
// A variation of it with a score weighting could be used to determine how similar two brackets are to each other in score space.
// The Gower Distance is also worth looking into.

use crate::bracket::Bracket;

#[derive(Debug, Clone)]
pub struct EvolvingPool{
    pub pool_size: i32, //how many bracket entries are you allowed to have in the pool
    pub brackets: Vec<Bracket>,
    pub batch: Batch, //the batch of brackets that the pool is currently being scored against
    pub batch_size: i32, //how many monte carlo simulations to run against each bracket in the pool
    pub mutation_rate: f64, //how often to mutate a bit in a child
    pub num_child_pools: i32, //how many child pools to create each generation
    pub max_score: f64, //the average maximum score of the best bracket for each round of scoring
    pub average_score: f64, //the average scores of all the brackets in the pool
    pub fitness: f64, //the fitness of the pool
}

//impl fn evolve, returns a new EvolvingPool?
//impl mutate to mutate every bracket in the pool
//impl create_n_child_pools to create n child pools with the parents of the current pool, returns a vector of EvolvingPools?
//impl fn score, finds the average maximum score of the best bracket for each round of scoring
//impl write_to_csv, writes the pool to a csv file
//impl pretty_print, prints the pool to the console

//impl mix_pool or breed_pool, takes two pools and mixes them together to create a new pool

//bracket impl calc_glower_distance, returns the glower distance between two brackets
//bracket impl calc_hamming_distance, returns the hamming distance between two brackets
//pool impl average_hamming_distance, returns the average hamming distance between all brackets in the pool

//find a way to detect convergence of the pool

impl EvolvingPool{
    //new function will create a pool of brackets from needed inputs
    pub fn new(tournamentinfo: &TournamentInfo, pool_size: i32, mutation_rate: f64, num_child_pools: i32, batch_size: i32) -> EvolvingPool{
        // create a vector of random brackets of size pool_size
        //let mut brackets: Vec<Bracket> = (0..pool_size).into_par_iter().map(|_| Bracket::new_from_binary( tournamentinfo, bracket::random63bool() )).collect();
        let brackets: Vec<Bracket> = (0..pool_size).into_par_iter().map(|_| Bracket::new( tournamentinfo)).collect();
        // let mut brackets: Vec<Bracket> = Vec::new();
        // for _i in 0..pool_size{
        //     brackets.push(Bracket::new(tournamentinfo));
        //     //let rand63bool = bracket::random63bool();
        //     //brackets.push(Bracket::new_from_binary(tournamentinfo, rand63bool));

        // }
        EvolvingPool{
            pool_size,
            brackets,
            batch: Batch::new(tournamentinfo, batch_size),
            batch_size,
            mutation_rate,
            num_child_pools,
            max_score: 0.0, //hasn't been scored yet
            average_score: 0.0, //hasn't been scored yet
            fitness: 0.0, //hasn't been scored yet
        }
    }
    pub fn score(&mut self, _tournamentinfo: &TournamentInfo) -> f64{
        let mut max_score = 0.0; //max average score of any one bracket in the batch
        let mut sum_score = 0.0; //sum of all the scores of the brackets in the batch
        for bracket in &mut self.brackets{
            let individual_score = self.batch.score_against_ref(bracket);
            sum_score += individual_score;
            if individual_score > max_score{
                max_score = individual_score;
            }
        }
        self.max_score = max_score;
        self.average_score = sum_score / self.pool_size as f64;
        let _average_ev = self.brackets.iter().map(|x| x.expected_value).sum::<f64>() / self.brackets.len() as f64 ;
        //self.fitness = self.average_score.powf(2.0) * self.hamming_distance_sum().sqrt().sqrt();
        //self.fitness = self.max_score;
        //self.fitness = self.brackets.iter().map(|x| x.expected_value).sum::<f64>() * self.brackets.iter().map(|x| x.prob).sum::<f64>();
        //self.fitness = self.hamming_distance_sum().sqrt() * self.average_score;
        //self.fitness = self.max_score * self.average_score.powf(3.0) * self.brackets.iter().map(|x| x.expected_value).sum::<f64>();;
        self.fitness = self.average_score;
        self.average_score
    }
    pub fn mutate(&self, tournamentinfo: &TournamentInfo) -> EvolvingPool{
        // mutate every bracket in the pool
        // Note: bracket.mutate() already creates a new bracket, so no need to clone first
        let brackets_to_mutate: Vec<Bracket> = self.brackets.iter()
            .map(|bracket| bracket.mutate(tournamentinfo, self.mutation_rate))
            .collect();

        EvolvingPool {
            pool_size: self.pool_size,
            brackets: brackets_to_mutate,
            batch: self.batch.clone(),
            batch_size: self.batch_size,
            mutation_rate: self.mutation_rate,
            num_child_pools: self.num_child_pools,
            max_score: 0.0,
            average_score: 0.0,
            fitness: 0.0,
        }
    }
    pub fn create_child_pools(&mut self, tournamentinfo: &TournamentInfo) -> Vec<EvolvingPool>{
        // create n child pools with the parents of the current pool
        let _child_pools: Vec<EvolvingPool> = Vec::new();
        let child_pools: Vec<EvolvingPool> = (0..self.num_child_pools).into_par_iter().map(|_| EvolvingPool::mutate(self, tournamentinfo)).collect();
        // //create n child pools
        // for _i in 0..self.num_child_pools{
        //     let mut child_pool = EvolvingPool::mutate(&self, tournamentinfo);
        //     child_pools.push(child_pool);
        // }
        // child_pools.push(self.clone()); //add the parent pool to the vector of child pools
        child_pools
    }
    pub fn update_batch(&mut self, tournamentinfo: &TournamentInfo){
        self.batch = Batch::new(tournamentinfo, self.batch_size);
    }
    pub fn pretty_print(&self, _tournamentinfo: &TournamentInfo){
        println!("Pool Size: {}", self.pool_size);
        println!("Mutation Rate: {}", self.mutation_rate);
        println!("Number of Child Pools: {}", self.num_child_pools);
        println!("Batch Size: {}", self.batch_size);
        println!("Max Score: {}", self.max_score);
        println!("Average Score: {}", self.average_score);
        println!("Fitness: {}", self.fitness);
        println!("Brackets:");
        for bracket in &self.brackets{
            bracket.pretty_print();
        }
    }
    /// Export bracket binary representations to a file.
    /// Since brackets contain RcTeam (not serializable), we export the binary representation
    /// which can be used to reconstruct brackets later.
    pub fn export_to_file(&self, _tournamentinfo: &TournamentInfo, filename: &str){
        let mut file = File::create(filename).unwrap();
        // Export just the binary representations (Vec<bool>) which are serializable
        let binaries: Vec<&Vec<bool>> = self.brackets.iter().map(|b| &b.binary).collect();
        let serialized = serde_json::to_string(&binaries).unwrap();
        file.write_all(serialized.as_bytes()).unwrap();
    }
    pub fn hamming_distance_sum(&mut self) -> f64{
        let mut sum = 0.0;
        for i in 0..self.brackets.len(){
            for j in 0..self.brackets.len(){
                sum += self.brackets[i].hamming_distance(&self.brackets[j]) as f64 / 63.0;
            }
        }
        sum / self.brackets.len() as f64
    }
}




#[derive(Debug, Clone)]
pub struct Batch{
    pub brackets: Vec<Bracket>,
    pub batch_score: f64,
    pub batch_score_std_dev: f64,
}

impl Batch{
    //This function will create a batch of brackets from MonteCarlo simulations
    pub fn new(tournamentinfo: &TournamentInfo, num_brackets: i32) -> Batch{
        let num_cpus = num_cpus::get();
        let num_brackets_per_core = num_brackets as usize / num_cpus;
        let brackets: Vec<Bracket> = (0..num_brackets).into_par_iter().with_min_len(num_brackets_per_core).map(|_| Bracket::new(tournamentinfo)).collect();
        Batch{
            brackets,
            batch_score: 0.0,
            batch_score_std_dev: 0.0,
        }
    }
    // This function will score each bracket in the batch against random sims and then return the average score
    pub fn score_against_simulations(&mut self,tournamentinfo: &TournamentInfo, num_sims: i32){
        //start sims
        for _i in 0..num_sims{
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
    pub fn score_against_ref(&mut self, ref_bracket: &Bracket) -> f64{
        //iterate through the brackets and score them against the reference bracket, collect it all into a new vector
        //doesn't need to be mutable
        let batch_scores: Vec<f64> = self.brackets.par_iter().map(|x| x.score(ref_bracket)).collect();
        //do the math to get the mean and standard deviation
        self.batch_score = batch_scores.iter().sum::<f64>() / batch_scores.len() as f64;
        self.batch_score_std_dev = ( batch_scores.iter().map(|x| (x - self.batch_score).powi(2) ).sum::<f64>() / batch_scores.len() as f64 ) .sqrt();
        //return the mean
        self.batch_score
    } 

}
