mod advancement;
mod anneal;
mod api;
mod bracket;
mod config;
mod elo;
mod exact;
mod field;
mod ga;
mod game_result;
mod ingest;
mod ncaa_bracket;
mod names;
mod optimize;
mod picks;
mod pool;
mod portfolio;
mod score;
mod torvik;
mod tree;

use bracket::{ScoringConfig, SeedScoring};
use clap::{Parser, ValueEnum};
use config::Config;
use ga::{
    GeneticAlgorithm, HybridOptimizer, LockSet, MonteCarloScenarios, SequentialPortfolioOptimizer,
    WholePortfolioGA,
};
use field::PickPopularity;
use portfolio::{AdvancementRound, BracketConstraint, BracketPortfolio, ConstrainedBracketBuilder};
use rand::SeedableRng;
use std::process::ExitCode;

#[derive(Debug, Clone, ValueEnum)]
enum DataSourceArg {
    /// BartTorvik season game log — one request, free, no key (default)
    Torvik,
    /// NCAA API (henrygd) scoreboard, fetched day by day
    Ncaa,
    /// ESPN API scoreboard, fetched day by day
    Espn,
    /// Frozen FiveThirtyEight CSV — 2023 only; 538 shut down in 2025
    Csv,
}

#[derive(Debug, Clone, ValueEnum)]
enum PortfolioStrategy {
    /// Greedy selection over exactly-optimal conditional brackets, then
    /// coordinate-ascent polish. Deterministic and strongest by best-ball.
    ExactBasis,
    /// Each bracket bets on a different championship winner (legacy)
    Champion,
    /// Greedily maximize EV while penalizing similarity (legacy)
    Diverse,
    /// Evolve whole portfolio at once using SA, fitness = best-ball score
    Annealing,
    /// Evolve whole portfolio at once using GA, fitness = best-ball score
    GaWhole,
    /// Sequential: optimize bracket 1, freeze, optimize bracket 2 for marginal contribution, etc.
    GaSequential,
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
enum ObjectiveArg {
    /// Expected best-ball score — right when the payout is linear in points
    BestBall,
    /// P(one of my entries finishes first) against the rest of the pool
    FirstPlace,
}

#[derive(Debug, Clone, ValueEnum)]
enum OptimizationMode {
    /// Exact dynamic-programming solve — provably optimal expected value
    Exact,
    /// Legacy hill-climbing optimization
    Legacy,
    /// Population-based Genetic Algorithm
    Ga,
    /// Hybrid Simulated Annealing + GA
    Hybrid,
}

#[derive(Parser, Debug)]
#[command(name = "ncaa-bracket-optimizer")]
#[command(author = "NCAA Bracket Optimizer")]
#[command(version = "1.0")]
#[command(about = "Optimizes March Madness brackets using ELO ratings and genetic algorithms")]
struct Args {
    /// Data source for team ratings
    #[arg(short, long, value_enum, default_value = "torvik")]
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

    /// Path to the frozen FiveThirtyEight CSV (only used with --source csv)
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
    #[arg(long, value_enum, default_value = "exact-basis")]
    portfolio_strategy: PortfolioStrategy,

    /// What the portfolio search maximizes (exact-basis strategy only)
    #[arg(long, value_enum, default_value = "best-ball")]
    objective: ObjectiveArg,

    /// Total entries in your pool, including yours. Sets how much competition
    /// `--objective first-place` optimizes against.
    #[arg(long, default_value = "100")]
    pool_entries: usize,

    /// Independent draws of the opposing field to average over
    #[arg(long, default_value = "8")]
    field_replicates: usize,

    /// JSON file of public pick rates (see --fetch-picks)
    #[arg(long)]
    pick_popularity: Option<String>,

    /// Fetch published pick rates from ESPN for this tournament year and cache them
    #[arg(long)]
    fetch_picks: Option<i32>,

    /// Fallback bias toward favourites when no pick data is available.
    /// 1.0 is the rating model itself; above that is a chalkier public.
    #[arg(long, default_value = "1.6")]
    chalk_tilt: f64,

    /// Number of steps for annealing strategy
    #[arg(long, default_value = "10000")]
    anneal_steps: usize,

    /// Lock a specific team to reach a round (format: "TeamName:FinalFour")
    /// Can be specified multiple times
    #[arg(long)]
    lock_team: Vec<String>,

    // ===== NEW GA OPTIONS =====

    /// Path to YAML configuration file
    #[arg(long)]
    config: Option<String>,

    /// Generate a sample configuration file and exit
    #[arg(long)]
    generate_config: bool,

    /// Optimization mode for single bracket
    #[arg(long, value_enum, default_value = "exact")]
    optimization_mode: OptimizationMode,

    /// Continue even when game data is missing or incomplete.
    /// Without this, an empty or partial season is a hard error rather than a
    /// run against uniform 1500 ratings that looks the same as a real one.
    #[arg(long, default_value = "false")]
    allow_partial_data: bool,

    /// GA population size (overrides config)
    #[arg(long)]
    population_size: Option<usize>,

    /// Simulation pool size for scoring (overrides config)
    #[arg(long)]
    pool_size: Option<usize>,

    /// Enable smart mutation (team/round based)
    #[arg(long)]
    smart_mutation: Option<bool>,

    /// Verbose output during optimization
    #[arg(short, long)]
    verbose: bool,

    // Scoring Configuration
    //
    // These override the `scoring:` section of the config file, which in turn
    // overrides the built-in defaults. They are Options so that "not passed"
    // is distinguishable from "passed the default value" — otherwise a config
    // file's scoring section can never take effect.

    /// Points for Round 1 [default: 1]
    #[arg(long)]
    score_r1: Option<f64>,
    /// Points for Round 2 [default: 2]
    #[arg(long)]
    score_r2: Option<f64>,
    /// Points for Round 3 / Sweet 16 [default: 4]
    #[arg(long)]
    score_r3: Option<f64>,
    /// Points for Round 4 / Elite 8 [default: 8]
    #[arg(long)]
    score_r4: Option<f64>,
    /// Points for Round 5 / Final Four [default: 16]
    #[arg(long)]
    score_r5: Option<f64>,
    /// Points for Round 6 / Championship [default: 32]
    #[arg(long)]
    score_r6: Option<f64>,

    /// Seed scoring mode for R1: add, multiply, or none [default: add]
    #[arg(long)]
    seed_r1: Option<String>,
    /// Seed scoring mode for R2 [default: add]
    #[arg(long)]
    seed_r2: Option<String>,
    /// Seed scoring mode for R3 [default: add]
    #[arg(long)]
    seed_r3: Option<String>,
    /// Seed scoring mode for R4 [default: multiply]
    #[arg(long)]
    seed_r4: Option<String>,
    /// Seed scoring mode for R5 [default: multiply]
    #[arg(long)]
    seed_r5: Option<String>,
    /// Seed scoring mode for R6 [default: multiply]
    #[arg(long)]
    seed_r6: Option<String>,
}

fn parse_seed_mode(s: &str) -> Result<SeedScoring, String> {
    match s.to_lowercase().as_str() {
        "add" => Ok(SeedScoring::Add),
        "multiply" | "mult" => Ok(SeedScoring::Multiply),
        "none" | "off" => Ok(SeedScoring::None),
        other => Err(format!(
            "unknown seed scoring mode '{}'. Expected add, multiply, or none",
            other
        )),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("\nError: {}", message);
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    if args.generate_config {
        println!("{}", config::generate_sample_config());
        println!("\n# Save this to config.yaml and customize as needed");
        return Ok(());
    }

    println!("NCAA Bracket Optimizer");
    println!("======================");
    println!();

    let mut app_config = Config::load_or_default(args.config.as_deref());
    if let Some(pop_size) = args.population_size {
        app_config.ga.population_size = pop_size;
    }
    if let Some(pool_size) = args.pool_size {
        app_config.simulation.pool_size = pool_size;
    }
    app_config.ga.generations = args.generations as usize;

    let scoring_config = scoring_from_args(&args, &app_config)?;
    print_scoring(&scoring_config);

    let tournamentinfo = match load_tournament(&args)? {
        Some(t) => t,
        None => return Ok(()), // --elo-only
    };

    let constraints = parse_lock_constraints(&args.lock_team)?;

    // Resolve constraints once, up front. A misspelled or ambiguous team name
    // and a pair of locks that cannot both hold are both errors here rather
    // than surprises after an optimization run.
    let mut builder = ConstrainedBracketBuilder::new(&tournamentinfo, &scoring_config);
    for constraint in &constraints {
        builder = builder.with_constraint(constraint.clone());
        println!(
            "Locking {} to reach {:?}",
            constraint.team_name, constraint.must_reach
        );
    }
    let locks = builder.locks()?;
    if !constraints.is_empty() {
        println!();
    }

    if let Some(num_brackets) = args.portfolio {
        run_portfolio_mode(
            &tournamentinfo,
            num_brackets,
            args.diversity_weight,
            &args.portfolio_strategy,
            &locks,
            args.generations,
            args.anneal_steps,
            &scoring_config,
            &app_config,
            &args,
            args.verbose,
        );
    } else {
        match args.optimization_mode {
            OptimizationMode::Exact => {
                run_exact_optimization(&tournamentinfo, &locks, &scoring_config)?
            }
            OptimizationMode::Legacy => run_legacy_optimization(
                &tournamentinfo,
                &locks,
                args.generations,
                args.batch_size,
                &scoring_config,
            ),
            OptimizationMode::Ga => {
                run_ga_optimization(&tournamentinfo, &app_config, &scoring_config, &locks, args.verbose)
            }
            OptimizationMode::Hybrid => run_hybrid_optimization(
                &tournamentinfo,
                &app_config,
                &scoring_config,
                &locks,
                args.verbose,
            ),
        }
    }

    Ok(())
}

/// Resolve scoring from built-in defaults, then the config file, then the CLI.
///
/// There used to be two of these live at once: `main` built one from the CLI
/// flags while the sequential-portfolio and hybrid optimizers built their own
/// from the YAML file. `--score-r6 100` therefore changed the answer in some
/// modes and was silently ignored in others.
fn scoring_from_args(args: &Args, app_config: &Config) -> Result<ScoringConfig, String> {
    let base = app_config
        .to_scoring_config()
        .map_err(|e| format!("invalid scoring in config file: {}", e))?;

    let cli_scores = [
        args.score_r1,
        args.score_r2,
        args.score_r3,
        args.score_r4,
        args.score_r5,
        args.score_r6,
    ];
    let cli_modes = [
        &args.seed_r1,
        &args.seed_r2,
        &args.seed_r3,
        &args.seed_r4,
        &args.seed_r5,
        &args.seed_r6,
    ];

    let mut resolved = base;
    for round in 0..6 {
        if let Some(points) = cli_scores[round] {
            resolved.round_scores[round] = points;
        }
        if let Some(mode) = cli_modes[round] {
            resolved.round_seed_scoring[round] = parse_seed_mode(mode)?;
        }
    }

    for (round, points) in resolved.round_scores.iter().enumerate() {
        if !points.is_finite() || *points < 0.0 {
            return Err(format!(
                "round {} score must be a non-negative number, got {}",
                round + 1,
                points
            ));
        }
    }

    Ok(resolved)
}

fn print_scoring(scoring_config: &ScoringConfig) {
    println!("Scoring Configuration:");
    for round in 0..6 {
        println!(
            "  R{}: {} ({:?})",
            round + 1,
            scoring_config.round_scores[round],
            scoring_config.round_seed_scoring[round]
        );
    }
    println!();
}

/// Load the tournament field, or `None` if `--elo-only` handled the run.
fn load_tournament(args: &Args) -> Result<Option<ingest::TournamentInfo>, String> {
    match args.source {
        DataSourceArg::Csv => {
            println!("Loading data from CSV file: {}", args.csv_path);
            println!(
                "Note: FiveThirtyEight shut down in 2025 and this file is a \
                 frozen 2023 snapshot. It is here for tests and benchmarks, \
                 not for picking a 2027 bracket."
            );
            Ok(Some(ingest::TournamentInfo::initialize(&args.csv_path)?))
        }
        DataSourceArg::Espn | DataSourceArg::Ncaa | DataSourceArg::Torvik => {
            let source = match args.source {
                DataSourceArg::Espn => api::DataSource::ESPN,
                DataSourceArg::Ncaa => api::DataSource::NCAA,
                DataSourceArg::Torvik => api::DataSource::Torvik,
                DataSourceArg::Csv => unreachable!(),
            };

            println!("Source: {:?}", source);
            println!("Season: {}", args.season);
            println!();

            let client = api::ApiClient::new(source, &args.cache_dir, args.allow_partial_data);

            println!("Fetching game data...");
            // A failed fetch used to fall back to an empty game list, which left
            // every team at the default 1500 rating. Every matchup then became a
            // coin flip and the optimizer produced confident-looking output from
            // pure noise, indistinguishable from a real run.
            let mut games = client.fetch_season(&args.season).map_err(|e| {
                format!(
                    "could not fetch game data: {}\n\
                     Ratings cannot be computed without games. Retry, or try \
                     another free source: `--source torvik`, `--source ncaa`.",
                    e
                )
            })?;

            println!();
            println!("Calculating ELO ratings from {} games...", games.len());
            let mut elo_system = elo::EloSystem::new(args.season.clone());
            elo_system.process_games(&mut games);

            if elo_system.games_processed == 0 && !args.allow_partial_data {
                return Err(format!(
                    "no completed games found for season {}. Ratings would all be \
                     the 1500 default, making every matchup a coin flip.\n\
                     Check the season string, or pass --allow-partial-data to proceed anyway.",
                    args.season
                ));
            }

            elo_system.print_top_teams(args.show_top);

            if args.elo_only {
                println!();
                println!("ELO-only mode: Skipping bracket optimization.");
                return Ok(None);
            }

            println!();
            let field = load_bracket_teams(args, &client)?;

            Ok(Some(ingest::TournamentInfo::from_elo_ratings_with_layout(
                &elo_system,
                field.teams,
                &field.region_layout,
            )?))
        }
    }
}

/// The four regions in the order `TournamentInfo` places them when the real
/// pairing is unknown — a bracket file or the sample field.
fn default_region_layout() -> [String; 4] {
    tree::REGION_ORDER.map(|r| r.to_string())
}

fn load_bracket_teams(
    args: &Args,
    client: &api::ApiClient,
) -> Result<ncaa_bracket::BracketField, String> {
    if let Some(ref bracket_path) = args.bracket_file {
        println!("Loading bracket from file: {}", bracket_path);
        let teams = api::load_bracket_from_file(bracket_path)?;
        println!("Loaded {} teams from bracket file", teams.len());
        // A bracket file carries no Final Four pairing, so the regions are
        // paired in the order they are listed.
        return Ok(ncaa_bracket::BracketField {
            teams,
            region_layout: default_region_layout(),
        });
    }

    // An explicit --tournament-year is a request for that specific bracket, so
    // failing to get it is an error. A year merely derived from the season is a
    // guess, and falling back to the sample field is reasonable there.
    if let Some(year) = args.tournament_year {
        println!("Fetching {} tournament bracket...", year);
        return client.fetch_tournament_bracket(year);
    }

    let derived_year = args
        .season
        .split('-')
        .nth(1)
        .and_then(|y| y.parse::<i32>().ok());

    match derived_year {
        Some(year) => {
            println!("Fetching {} tournament bracket (derived from season)...", year);
            match client.fetch_tournament_bracket(year) {
                Ok(field) => Ok(field),
                Err(e) => {
                    eprintln!("Note: {}", e);
                    println!("Using sample bracket teams");
                    Ok(ncaa_bracket::BracketField {
                        teams: ingest::TournamentInfo::sample_bracket_teams(),
                        region_layout: default_region_layout(),
                    })
                }
            }
        }
        None => {
            println!("Using sample bracket teams");
            Ok(ncaa_bracket::BracketField {
                teams: ingest::TournamentInfo::sample_bracket_teams(),
                region_layout: default_region_layout(),
            })
        }
    }
}

/// Parse team lock constraints from CLI arguments.
/// Format: "TeamName:Round".
fn parse_lock_constraints(lock_args: &[String]) -> Result<Vec<BracketConstraint>, String> {
    let mut constraints = Vec::new();

    for arg in lock_args {
        let (team_name, round_text) = arg
            .rsplit_once(':')
            .ok_or_else(|| format!("invalid --lock-team '{}', expected 'TeamName:Round'", arg))?;

        let round = match round_text.trim().to_lowercase().as_str() {
            "round2" | "r2" | "32" => AdvancementRound::Round2,
            "sweet16" | "s16" | "16" => AdvancementRound::Sweet16,
            "elite8" | "e8" | "8" => AdvancementRound::Elite8,
            "finalfour" | "f4" | "4" | "final4" => AdvancementRound::FinalFour,
            "championship" | "finals" | "2" => AdvancementRound::Championship,
            "winner" | "champion" | "1" => AdvancementRound::Winner,
            other => {
                return Err(format!(
                    "unknown round '{}' in --lock-team '{}'. Expected one of: \
                     Round2, Sweet16, Elite8, FinalFour, Championship, Winner",
                    other, arg
                ))
            }
        };

        constraints.push(BracketConstraint::new(team_name.trim(), round));
    }

    Ok(constraints)
}

/// Solve exactly for the highest-expected-value bracket.
fn run_exact_optimization(
    tournamentinfo: &ingest::TournamentInfo,
    locks: &LockSet,
    scoring_config: &ScoringConfig,
) -> Result<(), String> {
    println!();
    println!("=== Exact Optimization ===");
    println!("Maximizing expected score by dynamic programming over the bracket tree.");
    if !locks.is_empty() {
        println!("Subject to {} team lock(s).", locks.locks.len());
    }
    println!();

    let solution =
        exact::solve(tournamentinfo, scoring_config, &locks.locks).map_err(|e| e.to_string())?;

    solution.bracket.pretty_print();
    println!(
        "Optimal expected score: {:.2}  (no legal bracket scores higher under these rules)",
        solution.expected_value
    );

    Ok(())
}

/// Run GA-based single bracket optimization
fn run_ga_optimization(
    tournamentinfo: &ingest::TournamentInfo,
    app_config: &Config,
    scoring_config: &ScoringConfig,
    locks: &LockSet,
    verbose: bool,
) {
    println!();
    println!("=== Genetic Algorithm Optimization ===");
    println!("Population size: {}", app_config.ga.population_size);
    println!("Generations: {}", app_config.ga.generations);
    println!("Simulation pool size: {}", app_config.simulation.pool_size);
    println!();

    let pool = MonteCarloScenarios::new(
        tournamentinfo,
        app_config.simulation.pool_size,
        scoring_config,
    );

    let mut ga = GeneticAlgorithm::new(tournamentinfo, app_config.ga.clone(), *scoring_config)
        .with_locks(locks.clone(), tournamentinfo);

    let best_bracket = ga.run(tournamentinfo, &pool, verbose);

    println!();
    println!("Optimization complete!");
    println!();
    best_bracket.pretty_print();

    report_against_optimum(tournamentinfo, &best_bracket, scoring_config, locks);
}

/// Run hybrid SA+GA single bracket optimization
fn run_hybrid_optimization(
    tournamentinfo: &ingest::TournamentInfo,
    app_config: &Config,
    scoring_config: &ScoringConfig,
    locks: &LockSet,
    verbose: bool,
) {
    println!();
    println!("=== Hybrid SA+GA Optimization ===");
    println!("Generations: {}", app_config.ga.generations);
    println!("Simulation pool size: {}", app_config.simulation.pool_size);
    println!();

    let optimizer = HybridOptimizer::new(app_config.clone(), *scoring_config, locks.clone());
    let best_bracket = optimizer.optimize_single(tournamentinfo, verbose);

    println!();
    println!("Optimization complete!");
    println!();
    best_bracket.pretty_print();

    report_against_optimum(tournamentinfo, &best_bracket, scoring_config, locks);
}

/// Legacy hill-climbing: keep a champion bracket, mutate it, keep improvements.
fn run_legacy_optimization(
    tournamentinfo: &ingest::TournamentInfo,
    locks: &LockSet,
    generations: u32,
    batch_size: i32,
    scoring_config: &ScoringConfig,
) {
    println!();
    println!("=== Legacy Hill-Climbing Optimization ===");
    println!();

    let mut best = locks.repair(
        bracket::Bracket::new(tournamentinfo, Some(scoring_config)),
        tournamentinfo,
        scoring_config,
    );

    let mut batch = pool::Batch::new(tournamentinfo, batch_size, scoring_config);
    // The incumbent's fitness, measured the same way as every challenger's.
    // This used to be compared against `bracket.score` — the bracket's *perfect*
    // score, a constant several times larger than any batch average — so the
    // comparison never succeeded and the loop never accepted anything.
    let mut best_score = batch.score_against_ref(&best);

    let num_children = 63;
    let mut mutation_rate = 5.0 / 63.0;

    println!(
        "Starting {} generations, {} children per generation, initial mutation rate {:.4}",
        generations, num_children, mutation_rate
    );
    println!("Generation 0 score: {:.2}", best_score);

    for i in 0..generations {
        let children = best
            .create_n_children(tournamentinfo, num_children, mutation_rate, Some(scoring_config))
            .into_iter()
            .map(|c| locks.repair(c, tournamentinfo, scoring_config));

        for child in children {
            let score = batch.score_against_ref(&child);
            if score > best_score {
                best_score = score;
                best = child;
            }
        }

        let progress = i as f64 / generations as f64;
        if progress > 0.50 {
            mutation_rate = 1.0 / 63.0;
        } else if progress > 0.25 {
            mutation_rate = 2.0 / 63.0;
        }

        if i % 25 == 0 {
            println!(
                "{}: score {:.2}, std dev {:.2}, EV {:.2}",
                i, best_score, batch.batch_score_std_dev, best.expected_value
            );
        }
    }

    println!();
    println!("Optimization complete!");
    println!();
    best.pretty_print();

    report_against_optimum(tournamentinfo, &best, scoring_config, locks);
}

/// Show how far a search-based result falls short of the exact optimum.
///
/// The exact solve is cheap enough to run alongside any other mode, so there is
/// no reason to report a heuristic's score without the number it should be
/// compared against.
fn report_against_optimum(
    tournamentinfo: &ingest::TournamentInfo,
    bracket: &bracket::Bracket,
    scoring_config: &ScoringConfig,
    locks: &LockSet,
) {
    let found = exact::expected_value(tournamentinfo, bracket, scoring_config);
    println!("Expected score of this bracket: {:.2}", found);

    match exact::solve(tournamentinfo, scoring_config, &locks.locks) {
        Ok(solution) => {
            let gap = solution.expected_value - found;
            println!(
                "Exact optimum:                  {:.2}  (gap {:.2}, {:.1}% of optimum)",
                solution.expected_value,
                gap,
                100.0 * found / solution.expected_value
            );
        }
        Err(e) => eprintln!("Could not compute the exact optimum: {}", e),
    }
}

/// Resolve where the public's picks come from: an explicit file, a fetch from
/// ESPN, a previously cached fetch, or — failing all of those — a chalk model
/// standing in for data we do not have.
fn load_pick_popularity(
    tournamentinfo: &ingest::TournamentInfo,
    args: &Args,
) -> Result<PickPopularity, String> {
    if let Some(path) = &args.pick_popularity {
        return PickPopularity::load(path, tournamentinfo);
    }

    if let Some(year) = args.fetch_picks {
        let file = field::fetch_espn(year, &args.cache_dir)?;
        return PickPopularity::from_file(&file, tournamentinfo);
    }

    eprintln!(
        "No public pick data given (--pick-popularity or --fetch-picks); \n\
         falling back to a chalk model with tilt {:.2}. How concentrated the \n\
         public actually is on the favourite is the single biggest input to \n\
         this objective, so the real numbers are worth fetching.",
        args.chalk_tilt
    );
    Ok(PickPopularity::chalk(tournamentinfo, args.chalk_tilt))
}

/// Run portfolio mode - generate multiple diverse brackets
fn run_portfolio_mode(
    tournamentinfo: &ingest::TournamentInfo,
    num_brackets: usize,
    diversity_weight: f64,
    strategy: &PortfolioStrategy,
    locks: &LockSet,
    generations: u32,
    anneal_steps: usize,
    scoring_config: &ScoringConfig,
    app_config: &Config,
    args: &Args,
    verbose: bool,
) {
    println!();
    println!("=== Portfolio Mode ===");
    println!("Generating {} brackets...", num_brackets);
    println!("Strategy: {:?}", strategy);
    println!();

    let brackets: Vec<bracket::Bracket> = match strategy {
        PortfolioStrategy::ExactBasis => {
            println!("Mode: exact conditional basis + coordinate ascent");

            let pool = score::ScenarioPool::new(
                tournamentinfo,
                app_config.simulation.pool_size,
                scoring_config,
                ga::DEFAULT_POOL_SEED,
            );

            // The competition, if we are optimizing against it. Sampling the
            // field and reducing it to per-scenario order statistics is the
            // only up-front cost; after that an entry is evaluated at the same
            // price as under best-ball.
            let competition = match args.objective {
                ObjectiveArg::BestBall => None,
                ObjectiveArg::FirstPlace => {
                    let public = match load_pick_popularity(tournamentinfo, args) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            return;
                        }
                    };
                    let (favourite, share) = public.favourite();
                    println!("Public picks:  {}", public.source);
                    if let Some(n) = public.sample_size {
                        println!("Entries seen:  {}", n);
                    }
                    println!(
                        "Most-picked champion: {} ({:.1}% of entries)",
                        tournamentinfo.teams[favourite as usize].name,
                        share * 100.0
                    );

                    let opponents = args.pool_entries.saturating_sub(num_brackets).max(1);
                    println!(
                        "Competition:   {} opposing entries, averaged over {} draws of the field",
                        opponents, args.field_replicates
                    );
                    println!();

                    Some(pool.competition(
                        args.field_replicates.max(1),
                        opponents,
                        |replicate, i| {
                            let mut rng = rand::rngs::SmallRng::seed_from_u64(
                                ga::DEFAULT_POOL_SEED
                                    ^ ((replicate as u64) << 40)
                                    ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                            );
                            public.sample_entry(tournamentinfo, &mut rng)
                        },
                    ))
                }
            };

            let objective = match &competition {
                Some(c) => optimize::Objective::FirstPlace(c),
                None => optimize::Objective::BestBall,
            };
            println!("Maximizing:    {}", objective.name());
            println!(
                "Estimated against {} sampled tournaments",
                app_config.simulation.pool_size
            );
            println!();

            let plan = optimize::optimize_for(
                tournamentinfo,
                scoring_config,
                &pool,
                locks,
                num_brackets,
                objective,
                verbose,
            );

            println!("\n=== Portfolio Optimization Complete ===");
            println!("Conditionally optimal brackets considered: {}", plan.basis_size);
            println!("Coordinate-ascent sweeps to convergence:   {}", plan.sweeps);

            match &competition {
                None => {
                    // Optimizing against a finite sample always flatters
                    // itself; a second, independently drawn pool says how much
                    // of the gain survives outside the sample it was fitted to.
                    let holdout = optimize::holdout_score(
                        tournamentinfo,
                        scoring_config,
                        &plan.entries,
                        app_config.simulation.pool_size.max(50_000),
                        ga::DEFAULT_POOL_SEED ^ 0xFFFF_FFFF,
                    );
                    println!("Best single entry:        {:.2}", plan.single_entry);
                    println!("Best-ball, in-sample:     {:.2}", plan.best_ball);
                    println!("Best-ball, held out:      {:.2}", holdout);
                    println!("Added by the local search: {:+.2}", plan.polish_gain);
                }
                Some(c) => {
                    println!("Top opposing score, average: {:.1}", c.mean_top_score());
                    println!(
                        "P(finish first) with 1 entry:  {:.3}%",
                        plan.single_entry * 100.0
                    );
                    println!(
                        "P(finish first) with {} entries: {:.3}%  ({:+.3}% from the local search)",
                        num_brackets,
                        plan.best_ball * 100.0,
                        plan.polish_gain * 100.0
                    );
                    let fair = num_brackets as f64 / args.pool_entries.max(1) as f64;
                    println!(
                        "A random entrant's share would be {:.3}%, so this is {:.2}x fair",
                        fair * 100.0,
                        plan.best_ball / fair
                    );
                }
            }
            if num_brackets > 1 {
                println!(
                    "Entries differ on {:.0} of 63 games on average",
                    plan.mean_spread()
                );
            }

            plan.brackets(tournamentinfo, scoring_config)
        }
        PortfolioStrategy::Champion => {
            let portfolio =
                BracketPortfolio::generate_champion_stratified(tournamentinfo, num_brackets, scoring_config);
            portfolio.print_summary_with_config(Some(scoring_config));
            portfolio.print_pairwise_distances(scoring_config);
            portfolio.brackets
        }
        PortfolioStrategy::Diverse => {
            println!("Diversity weight: {:.2}", diversity_weight);
            let portfolio = BracketPortfolio::generate_greedy_diverse(
                tournamentinfo,
                num_brackets,
                diversity_weight,
                generations,
                scoring_config,
            );
            portfolio.print_summary_with_config(Some(scoring_config));
            portfolio.print_pairwise_distances(scoring_config);
            portfolio.brackets
        }
        PortfolioStrategy::Annealing => {
            println!("Mode: Simulated Annealing (whole portfolio)");
            println!(
                "Fitness: Best-ball score against {} simulations",
                app_config.simulation.pool_size
            );
            println!("Annealing steps: {}", anneal_steps);
            println!();
            let portfolio = BracketPortfolio::generate_annealing_diverse(
                tournamentinfo,
                num_brackets,
                app_config.simulation.pool_size,
                anneal_steps,
                scoring_config,
            );
            portfolio.print_summary_with_config(Some(scoring_config));
            portfolio.brackets
        }
        PortfolioStrategy::GaWhole => {
            println!("Mode: Genetic Algorithm (whole portfolio evolution)");
            println!("Population size: {} portfolios", app_config.ga.population_size);
            println!("Generations: {}", app_config.ga.generations);
            println!(
                "Fitness: Best-ball score against {} simulations",
                app_config.simulation.pool_size
            );
            println!();

            let pool = MonteCarloScenarios::new(
                tournamentinfo,
                app_config.simulation.pool_size,
                scoring_config,
            );

            let mut ga = WholePortfolioGA::new(
                tournamentinfo,
                num_brackets,
                app_config.ga.clone(),
                *scoring_config,
            )
            .with_locks(locks.clone(), tournamentinfo);

            let brackets = ga.run(tournamentinfo, &pool, verbose);

            println!("\n=== Portfolio Optimization Complete ===");
            println!(
                "Final best-ball score: {:.2}",
                pool.score_portfolio_best_ball(&brackets, scoring_config)
            );
            brackets
        }
        PortfolioStrategy::GaSequential => {
            println!("Mode: Sequential GA (freeze-and-optimize)");
            println!("Population size: {}", app_config.ga.population_size);
            println!("Generations per bracket: {}", app_config.ga.generations);
            println!("Simulation pool size: {}", app_config.simulation.pool_size);
            println!();

            let optimizer =
                SequentialPortfolioOptimizer::new(app_config.clone(), *scoring_config, locks.clone());
            optimizer.optimize(tournamentinfo, num_brackets, verbose)
        }
    };

    println!("\nPortfolio champions:");
    for (i, b) in brackets.iter().enumerate() {
        println!(
            "  Bracket {}: {} (seed {}) - expected score {:.2}",
            i + 1,
            b.winner.name,
            b.winner.seed,
            exact::expected_value(tournamentinfo, b, scoring_config)
        );
    }

    for (i, b) in brackets.iter().enumerate() {
        println!("\n=== Bracket {} ===", i + 1);
        b.pretty_print();
    }
}
