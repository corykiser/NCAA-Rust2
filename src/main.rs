mod ingest;
mod bracket;
mod pool;
use rand::Rng;
fn main() {
    
    let tournamentinfo = ingest::TournamentInfo::initialize();

    let ref_bracket = bracket::Bracket::new(&tournamentinfo);

    let mut random_63_bool: Vec<bool> = Vec::new();
    for _i in 0..63 {
        let mut rng = rand::thread_rng();
        let rand_bool: bool = rng.gen(); // generates a float between 0 and 1
        random_63_bool.push(rand_bool);
    }
    println!("{:?}", random_63_bool);

    let generated_bracket = bracket::Bracket::new_from_binary(&tournamentinfo, random_63_bool);
    
    println!("{:?}", generated_bracket);

}
