mod ingest;
mod bracket;
mod pool;
mod elo;
mod api;
mod game_result;
mod portfolio;
mod genetic;

use clap::{Parser, ValueEnum};
use rand::Rng;
use portfolio::{BracketPortfolio, BracketConstraint, AdvancementRound, ConstrainedBracketBuilder, GeneticMode};

#[derive(Debug, Clone, ValueEnum)]
enum DataSourceArg {
    /// Use ESPN API for live game data
    Espn,
    /// Use NCAA API (henrygd) for live game data
    Ncaa,
    /// Use FiveThirtyEight CSV file (legacy)
    Csv,
}

#[derive(Debug, Clone, ValueEnum)]
enum PortfolioStrategy {
    /// Each bracket bets on a different championship winner
    Champion,
    /// Greedily maximize EV while penalizing similarity to previous brackets
    Diverse,
    /// Optimize for maximum portfolio performance using genetic algorithm
    Genetic,
}

#[derive(Parser, Debug)]
#[command(name = "ncaa-bracket-optimizer")]
#[command(author = "NCAA Bracket Optimizer")]
#[command(version = "1.0")]
#[command(about = "Optimizes March Madness brackets using ELO ratings and genetic algorithms")]
struct Args {
    /// Data source for team ratings
    #[arg(short, long, value_enum, default_value = "espn")]
    source: DataSourceArg,

    /// Season to analyze (e.g., 2024-2025)
    #[arg(long, default_value_t = api::current_season())]
    season: String,

    /// Tournament year (e.g., 2024 for March Madness 2024)
    /// Used to fetch bracket teams with seeds and regions
    #[arg(short, long)]
    tournament_year: Option<i32>,

    /// Path to bracket JSON file with teams, seeds, and regions
    /// Alternative to fetching from API
    #[arg(long)]
    bracket_file: Option<String>,

    /// Path to FiveThirtyEight CSV file (only used with --source csv)
    #[arg(long, default_value = "fivethirtyeight_ncaa_forecasts.csv")]
    csv_path: String,

    /// Cache directory for API data
    #[arg(long, default_value = "./data")]
    cache_dir: String,

    /// Number of generations for genetic algorithm
    #[arg(short, long, default_value = "200")]
    generations: u32,

    /// Batch size for scoring simulations
    #[arg(short, long, default_value = "1000")]
    batch_size: i32,

    /// Show top N teams by ELO rating
    #[arg(long, default_value = "25")]
    show_top: usize,

    /// Only calculate and show ELO ratings (skip bracket optimization)
    #[arg(long, default_value = "false")]
    elo_only: bool,

    /// Generate a portfolio of N diverse brackets instead of a single optimized bracket
    #[arg(long)]
    portfolio: Option<usize>,

    /// Diversity weight for portfolio generation (higher = more diverse brackets)
    #[arg(long, default_value = "5.0")]
    diversity_weight: f64,

    /// Strategy for portfolio generation
    #[arg(long, value_enum, default_value = "champion")]
    portfolio_strategy: PortfolioStrategy,

    /// Lock a specific team to reach a round (format: "TeamName:FinalFour")
    /// Can be specified multiple times
    #[arg(long)]
    lock_team: Vec<String>,

    /// Number of simulations for genetic portfolio optimization
    #[arg(long, default_value = "10000")]
    genetic_sims: i32,

    /// Mode for genetic algorithm
    #[arg(long, value_enum, default_value = "sequential")]
    genetic_mode: GeneticMode,
}

fn main() {
    let args = Args::parse();

    println!("NCAA Bracket Optimizer");
    println!("======================");
    println!();

    let tournamentinfo = match args.source {
        DataSourceArg::Csv => {
            println!("Loading data from CSV file: {}", args.csv_path);
            ingest::TournamentInfo::initialize(&args.csv_path)
        }
        DataSourceArg::Espn | DataSourceArg::Ncaa => {
            let source = match args.source {
                DataSourceArg::Espn => api::DataSource::ESPN,
                DataSourceArg::Ncaa => api::DataSource::NCAA,
                _ => unreachable!(),
            };

            println!("Source: {:?}", source);
            println!("Season: {}", args.season);
            println!();

            // Initialize API client
            let client = api::ApiClient::new(source, &args.cache_dir);

            // Fetch games for the season
            println!("Fetching game data...");
            let mut games = match client.fetch_season(&args.season) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Error fetching games: {}", e);
                    eprintln!("Falling back to sample bracket with default ratings...");
                    Vec::new()
                }
            };

            // Calculate ELO ratings
            println!();
            println!("Calculating ELO ratings from {} games...", games.len());
            let mut elo_system = elo::EloSystem::new(args.season.clone());
            elo_system.process_games(&mut games);

            // Show top teams
            elo_system.print_top_teams(args.show_top);

            if args.elo_only {
                println!();
                println!("ELO-only mode: Skipping bracket optimization.");
                return;
            }

            // Get bracket teams - try multiple sources in order:
            // 1. Local bracket file (if provided)
            // 2. Fetch from API (if tournament_year provided)
            // 3. Fall back to sample data
            println!();
            let bracket_teams = if let Some(ref bracket_path) = args.bracket_file {
                println!("Loading bracket from file: {}", bracket_path);
                match api::load_bracket_from_file(bracket_path) {
                    Ok(teams) => {
                        println!("Loaded {} teams from bracket file", teams.len());
                        teams
                    }
                    Err(e) => {
                        eprintln!("Error loading bracket file: {}", e);
                        eprintln!("Falling back to sample bracket...");
                        ingest::TournamentInfo::sample_bracket_teams()
                    }
                }
            } else if let Some(year) = args.tournament_year {
                println!("Fetching {} tournament bracket...", year);
                match client.fetch_tournament_bracket(year) {
                    Ok(teams) => teams,
                    Err(e) => {
                        eprintln!("Error fetching bracket: {}", e);
                        eprintln!("Falling back to sample bracket...");
                        ingest::TournamentInfo::sample_bracket_teams()
                    }
                }
            } else {
                // Derive tournament year from season
                let parts: Vec<&str> = args.season.split('-').collect();
                if parts.len() == 2 {
                    if let Ok(year) = parts[1].parse::<i32>() {
                        println!("Fetching {} tournament bracket (derived from season)...", year);
                        match client.fetch_tournament_bracket(year) {
                            Ok(teams) => teams,
                            Err(e) => {
                                eprintln!("Note: {}", e);
                                println!("Using sample bracket teams");
                                ingest::TournamentInfo::sample_bracket_teams()
                            }
                        }
                    } else {
                        println!("Using sample bracket teams");
                        ingest::TournamentInfo::sample_bracket_teams()
                    }
                } else {
                    println!("Using sample bracket teams");
                    ingest::TournamentInfo::sample_bracket_teams()
                }
            };

            // Create tournament info with calculated ELO ratings
            ingest::TournamentInfo::from_elo_ratings(&elo_system, bracket_teams)
        }
    };

    // Parse any team lock constraints
    let constraints = parse_lock_constraints(&args.lock_team);

    // Check if we should run in portfolio mode
    if let Some(num_brackets) = args.portfolio {
        run_portfolio_mode(
            &tournamentinfo,
            num_brackets,
            args.diversity_weight,
            &args.portfolio_strategy,
            &constraints,
            args.generations,
            args.genetic_sims,
            args.genetic_mode,
        );
    } else if !constraints.is_empty() {
        // Run single bracket with constraints
        run_constrained_optimization(&tournamentinfo, &constraints, args.generations, args.batch_size);
    } else {
        // Run standard bracket optimization
        run_optimization(&tournamentinfo, args.generations, args.batch_size);
    }
}

/// Parse team lock constraints from CLI arguments
/// Format: "TeamName:Round" where Round is one of:
/// Round2, Sweet16, Elite8, FinalFour, Championship, Winner
fn parse_lock_constraints(lock_args: &[String]) -> Vec<BracketConstraint> {
    let mut constraints = Vec::new();

    for arg in lock_args {
        let parts: Vec<&str> = arg.split(':').collect();
        if parts.len() != 2 {
            eprintln!("Warning: Invalid lock format '{}', expected 'TeamName:Round'", arg);
            continue;
        }

        let team_name = parts[0].trim();
        let round = match parts[1].trim().to_lowercase().as_str() {
            "round2" | "r2" | "32" => AdvancementRound::Round2,
            "sweet16" | "s16" | "16" => AdvancementRound::Sweet16,
            "elite8" | "e8" | "8" => AdvancementRound::Elite8,
            "finalfour" | "f4" | "4" | "final4" => AdvancementRound::FinalFour,
            "championship" | "finals" | "2" => AdvancementRound::Championship,
            "winner" | "champion" | "1" => AdvancementRound::Winner,
            other => {
                eprintln!("Warning: Unknown round '{}', skipping", other);
                continue;
            }
        };

        constraints.push(BracketConstraint::new(team_name, round));
        println!("Locking {} to reach {:?}", team_name, round);
    }

    constraints
}

/// Run portfolio mode - generate multiple diverse brackets
fn run_portfolio_mode(
    tournamentinfo: &ingest::TournamentInfo,
    num_brackets: usize,
    diversity_weight: f64,
    strategy: &PortfolioStrategy,
    constraints: &[BracketConstraint],
    generations: u32,
    genetic_sims: i32,
    genetic_mode: GeneticMode,
) {
    println!();
    println!("=== Portfolio Mode ===");
    println!("Generating {} diverse brackets...", num_brackets);
    println!("Strategy: {:?}", strategy);
    println!("Diversity weight: {:.2}", diversity_weight);

    if !constraints.is_empty() {
        println!("Base constraints:");
        for c in constraints {
            println!("  - {} must reach {:?}", c.team_name, c.must_reach);
        }
    }
    println!();

    let portfolio = match strategy {
        PortfolioStrategy::Champion => {
            BracketPortfolio::generate_champion_stratified(tournamentinfo, num_brackets)
        }
        PortfolioStrategy::Diverse => {
            BracketPortfolio::generate_greedy_diverse(
                tournamentinfo,
                num_brackets,
                diversity_weight,
                generations,
            )
        }
        PortfolioStrategy::Genetic => {
            BracketPortfolio::generate_genetic_portfolio(
                tournamentinfo,
                num_brackets,
                genetic_sims,
                generations,
                genetic_mode,
            )
        }
    };

    portfolio.print_summary();
    portfolio.print_pairwise_distances();

    // Print each bracket in detail
    for (i, bracket) in portfolio.brackets.iter().enumerate() {
        println!("\n=== Bracket {} ===", i + 1);
        bracket.pretty_print();
    }
}

/// Run optimization with locked team constraints
fn run_constrained_optimization(
    tournamentinfo: &ingest::TournamentInfo,
    constraints: &[BracketConstraint],
    generations: u32,
    batch_size: i32,
) {
    println!();
    println!("=== Constrained Bracket Mode ===");
    println!("Building bracket with {} constraints...", constraints.len());
    println!();

    // Build initial bracket with constraints
    let mut builder = ConstrainedBracketBuilder::new(tournamentinfo);
    for constraint in constraints {
        builder = builder.with_constraint(constraint.clone());
    }

    match builder.build() {
        Ok(mut bracket) => {
            println!("Initial constrained bracket built.");
            println!("Champion: {} (seed {})", bracket.winner.name, bracket.winner.seed);
            println!("Expected Value: {:.2}", bracket.expected_value);
            println!();

            // Now optimize around the constraints
            println!("Optimizing bracket with constraints...");
            let mut max_bracket = bracket.clone();
            let mutation_rate = 1.0 / 63.0 * 3.0;

            // Create batch for scoring
            let mut batch = pool::Batch::new(tournamentinfo, batch_size);

            for i in 0..generations {
                // Create mutated child
                let child = max_bracket.mutate(tournamentinfo, mutation_rate);

                // Score against batch
                batch.score_against_ref(&child);

                if batch.batch_score > bracket.score {
                    max_bracket = child;
                }

                if i % 50 == 0 {
                    println!("Generation {}: EV = {:.2}", i, max_bracket.expected_value);
                }
            }

            println!();
            println!("Optimization complete!");
            println!();
            max_bracket.pretty_print();
        }
        Err(e) => {
            eprintln!("Error building constrained bracket: {}", e);
        }
    }
}

fn run_optimization(tournamentinfo: &ingest::TournamentInfo, generations: u32, batch_size: i32) {
    let mut random_63_bool: Vec<bool> = Vec::new();
    for _i in 0..63 {
        let mut rng = rand::thread_rng();
        let rand_bool: bool = rng.gen();
        random_63_bool.push(rand_bool);
    }

    // Start with a bracket that is a likely scenario
    let generated_bracket = bracket::Bracket::new(tournamentinfo);

    let num_children = 63;
    let mut mutation_rate = 1.0 / 63.0 * 5.0;

    let mut max_score = 0.0;
    let mut max_std_dev = 0.0;
    let mut max_bracket = generated_bracket.clone();

    // Create a batch of random brackets to score against
    let mut generated_batch = pool::Batch::new(tournamentinfo, batch_size);

    // For tracking the moving average
    let mut moving_average_tracker: Vec<f64> = Vec::new();

    // For tracking if the fittest individual is changing from generation to generation
    let _last_max_bracket = generated_bracket.clone();

    println!();
    println!("Starting bracket optimization...");
    println!();

    for i in 0..generations {
        // Show the score of the random bracket before any optimization
        if i == 0 {
            println!("Starting {} generations of optimization, with {} children per generation, and a mutation rate of {:.4}",
                     generations, num_children, mutation_rate);
            println!();
            generated_batch.score_against_ref(&generated_bracket);
            println!("{}, The score of the original bracket is: {:.2} std_dev: {:.2}",
                     i, generated_batch.batch_score, generated_batch.batch_score_std_dev);
        }

        // Create a batch of random brackets to score against for this round
        let mut generated_batch = pool::Batch::new(tournamentinfo, batch_size);

        // Score the batch against the generated bracket
        let mut children = max_bracket.create_n_children(tournamentinfo, num_children, mutation_rate);

        let last_max_bracket = max_bracket.clone();

        // Score each of the children against the batch, then select the best child
        let mut fitness = 0.0;
        for child in &mut children {
            generated_batch.score_against_ref(child);
            if generated_batch.batch_score > fitness {
                fitness = generated_batch.batch_score;
                max_score = generated_batch.batch_score;
                max_std_dev = generated_batch.batch_score_std_dev;
                max_bracket = child.clone();
            }
        }

        // Test for change from generation to generation of the fittest individual
        let same_flag = last_max_bracket == max_bracket;

        // Mutation rate should decrease over time
        if i as f64 / generations as f64 > 0.25 {
            mutation_rate /= 2.0;
        }
        if i as f64 / generations as f64 > 0.50 {
            mutation_rate = 1.0 / 63.0;
        }

        if i % 1 == 0 {
            moving_average_tracker.push(max_score);
            if moving_average_tracker.len() > 10 {
                moving_average_tracker.remove(0);
            }
            let moving_average: f64 = moving_average_tracker.iter().sum::<f64>() / moving_average_tracker.len() as f64;
            println!("{}, The average score so far is: {:.2}, std_dev: {:.2}, moving average: {:.2}, ev: {:.2}, same: {}",
                     i, max_score, max_std_dev, moving_average, max_bracket.expected_value, same_flag);
        }
    }

    println!();
    println!("Optimization complete!");
    println!();
    max_bracket.pretty_print();
}
