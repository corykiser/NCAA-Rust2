mod ingest;
mod bracket;
mod pool;
fn main() {
    
    let tournamentinfo = ingest::TournamentInfo::initialize();

    let ref_bracket = bracket::Bracket::new(&tournamentinfo);
    let mut score: f64 = 0.0;

    let num_sims = 10000;


    for i in 0..num_sims {
        let bracket = bracket::Bracket::new(&tournamentinfo);
        score += bracket.score(&ref_bracket);
        if i % 100 == 0 {
            println!("{} {}", i, score / (i as f64));
        }

        //bracket.winner.print();
    }
    println!("{}", ref_bracket.expected_value * num_sims as f64);

}
