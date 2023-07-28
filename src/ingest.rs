// This file is used to ingest the data from the csv file that contains the 538 ratings and store it in a struct to be used later in other parts of the program

use serde::{Serialize, Deserialize};
use csv;

use csv::StringRecord;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Team {
    pub name: String,
    pub seed: i32,
    pub region: String,
    pub rating: f32,
}

impl Team {
    pub fn new(name: String, seed: i32, region: String, rating: f32) -> Team {
        Team {
            name,
            seed,
            region,
            rating,
        }
    }
    pub fn print(&self) {
        println!("{:?}", self);
    }

}
impl PartialEq for Team {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TournamentInfo {
    pub teams: Vec<Team>,
    pub round1: [[i32; 2]; 8],
    pub round2: [[i32; 4]; 4],
    pub round3: [[i32; 8]; 2],
    pub round4: [[i32; 16]; 1],
    pub regions: Vec<Vec<Team>>,
}

impl TournamentInfo {
    // Inititialize the vector of teams to be hold 64 teams
    pub fn initialize(file_path: &str) -> TournamentInfo {
        //structure of tournament: What seed plays the other seeds in each round?
        let round1: [[i32; 2]; 8] = [
            [1, 16],
            [2, 15],
            [3, 14],
            [4, 13],
            [5, 12],
            [6, 11],
            [7, 10],
            [8, 9],
        ];
        let round2: [[i32; 4]; 4] = [
            [1, 16, 8, 9],
            [5, 12, 4, 13],
            [6, 11, 3, 14],
            [7, 10, 2, 15],
        ];
        let round3: [[i32; 8]; 2] = [[1, 16, 8, 9, 5, 12, 4, 13], [6, 11, 3, 14, 7, 10, 2, 15]];
        let round4: [[i32; 16]; 1] = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]];

        let mut teams: Vec<Team> = Vec::with_capacity(64);

        //TODO check if file exists and download if it doesn't exist OR specify file path as an argument
        //let file_path = "/Users/corydkiser/Documents/ncaa/fivethirtyeight_ncaa_forecasts.csv";
        let mut rdr = csv::Reader::from_path(file_path).expect("file access error");
        let mut mensrecords: Vec<StringRecord> = Vec::new(); //holds the csv records

        // Loop over each record.
        for result in rdr.records() {
            let record = &result.unwrap();
            if record[0].starts_with("mens")
                && record[1].contains("2023-03-15")
                //TODO find the latest date with the last exactly 64 teams (there are 64 record[3] entries that equal 1) to sort out the play in games
                && record[3].contains("1.0")
            {
                mensrecords.push(record.clone());
            }
        }

        //create containers
        let mut rating: [f64; 64] = [0.0; 64];
        let mut teamid: [u32; 64] = [0; 64];
        let mut seed: [u32; 64] = [0; 64];
        let mut name: Vec<String> = Vec::new();
        let mut region: Vec<String> = Vec::new();

        for i in 0..mensrecords.len() {
            rating[i] = mensrecords[i][14].parse().unwrap(); //populate Team Ratings array
            teamid[i] = mensrecords[i][12].parse().unwrap(); //populate team id array
            name.push(mensrecords[i][13].to_string()); //populate team name array
            region.push(mensrecords[i][15].to_string()); //populate regions array
            //below removes non ascii digits "a" and "b" from team seeds
            //let test = "12b3as>";
            //let test2: String = test.to_string().chars().filter(|x| x.is_ascii_digit()).collect();
            if mensrecords[i][16].ends_with("a") || mensrecords[i][16].ends_with("b") {
                //let length = mensrecords[i][16].len();
                let mut tempstring = mensrecords[i][16].to_string();
                tempstring.pop();
                seed[i] = tempstring.parse().unwrap(); //populate team seed array
            } else {
                seed[i] = mensrecords[i][16].parse().unwrap();
            }
            //add team to the vector
            teams.push(Team::new(name[i].clone(), seed[i] as i32, region[i].clone(), rating[i] as f32));
            //add team to the region hashset

        }
        assert!(teams.len() == 64, "There are not 64 teams in the tournament");

        let east: Vec<Team> = teams
        .iter()
        .filter(|&x| x.region == "East")
        .cloned()
        .collect();

        let west: Vec<Team> = teams
        .iter()
        .filter(|&x| x.region == "West")
        .cloned()
        .collect();

        let south: Vec<Team> = teams
        .iter()
        .filter(|&x| x.region == "South")
        .cloned()
        .collect();

        let midwest: Vec<Team> = teams
        .iter()
        .filter(|&x| x.region == "Midwest")
        .cloned()
        .collect();

        let regions = vec![east, west, south, midwest];


        TournamentInfo { 
            teams: teams, 
            round1: round1, 
            round2: round2, 
            round3: round3, 
            round4: round4,
            regions: regions,
        }
    }
    pub fn print(&self) {
        println!("{:?}", self);
    }
}