mod ingest;
mod bracket;
mod pool;


use rand::Rng;
fn main() {

    let file_path = "/Users/corydkiser/Documents/ncaa/fivethirtyeight_ncaa_forecasts.csv";
    
    let tournamentinfo = ingest::TournamentInfo::initialize(file_path);

    //let ref_bracket = bracket::Bracket::new(&tournamentinfo);

    let mut random_63_bool: Vec<bool> = Vec::new();
    for _i in 0..63 {
        let mut rng = rand::thread_rng();
        let rand_bool: bool = rng.gen(); // generates a float between 0 and 1
        random_63_bool.push(rand_bool);
    }
    println!("{:?}", random_63_bool);
    
    //uncoment to start totally random
    //let mut generated_bracket = bracket::Bracket::new_from_binary(&tournamentinfo, random_63_bool.clone());

    //start with a bracket that is a likely scenario
    let generated_bracket = bracket::Bracket::new(&tournamentinfo);

    let generations = 200;
    let num_children = 63; //I think 63 is helpful? I'm not sure
    let mut mutation_rate = 1.0/63.0 * 5.0; //I think 1/63 is helpful? I'm not sure. One bit flip per child on average.
    let batch_size = 1000; //increase]ing this uses more cores bc of how rayon works, obviously it is more compute though


    let mut max_score = 0.0;
    let mut max_std_dev = 0.0;
    let mut max_bracket = generated_bracket.clone();
    //create a batch of random brackets to score against
    let mut generated_batch = pool::Batch::new(&tournamentinfo, batch_size);

    //for tracking the moving average
    let mut moving_average_tracker: Vec<f64> = Vec::new();

    //for tracking if the fittest individual is changing from generation to generation
    let _last_max_bracket = generated_bracket.clone();
    
    for i in 0..generations{

        //show the score of the random bracket before any optimization
        if i == 0{
            println!();
            println!("Starting {} generations of optimization, with {} children per generation, and a mutation rate of {}", generations, num_children, mutation_rate);
            println!("");
            generated_batch.score_against_ref(&generated_bracket);
            println!("{}, The score of the original bracket is: {} std_dev: {}", i, generated_batch.batch_score, generated_batch.batch_score_std_dev);
        }

        //create a batch of random brackets to the new round score against
        let mut generated_batch = pool::Batch::new(&tournamentinfo, batch_size);

        //score the batch against the generated bracket (aka score the generated bracket against the batch)
        let mut children = max_bracket.create_n_children(&tournamentinfo, num_children, mutation_rate); //this adds the parent back in for n+1 children total

        let last_max_bracket = max_bracket.clone();


        //score each of the children against the batch, then select the best child
        let mut fitness = 0.0; //reset max score of batch
        for child in &mut children{
            generated_batch.score_against_ref(&child);
            if generated_batch.batch_score > fitness{
            //if generated_batch.batch_score.powf(8.0) * (1.0 / generated_batch.batch_score_std_dev) > fitness{
                fitness = generated_batch.batch_score;
                //fitness = generated_batch.batch_score.powf(8.0) * (1.0 / generated_batch.batch_score_std_dev);
                max_score = generated_batch.batch_score;
                max_std_dev = generated_batch.batch_score_std_dev;
                max_bracket = child.clone();
            }
        }
        
        //test for change from generation to generation of the fittest individual
        let same_flag = if last_max_bracket == max_bracket{
            true
        } else {
            false
        };

        //mutation rate should decrease over time
        if i as f64 / generations as f64 > 0.25{
            mutation_rate = mutation_rate / 2.0;
        }
        if i as f64 / generations as f64 > 0.50{
            mutation_rate = 1.0 / 63.0; //only single bit flips on average
        }

        if i % 1 == 0{
            moving_average_tracker.push(max_score);
            if moving_average_tracker.len() > 10{
                moving_average_tracker.remove(0);
            }
            let moving_average: f64 = moving_average_tracker.iter().sum::<f64>() / moving_average_tracker.len() as f64;
            println!("{}, The average score so far is: {:.2}, std_dev: {:.2}, moving average: {:.2}, ev: {:.2}, same: {}", i, max_score, max_std_dev, moving_average, max_bracket.expected_value, same_flag);
        }
        
    }
    max_bracket.pretty_print();

    
}
