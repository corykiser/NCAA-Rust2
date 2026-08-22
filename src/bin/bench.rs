//! Benchmark harness for the optimizer hot paths.
//!
//! Deliberately independent of the network data sources: it builds a synthetic
//! 64-team field with a realistic rating spread so runs are reproducible and
//! comparable across commits.
//!
//! ```text
//! cargo run --release --bin bench -- [filter]
//! ```

use std::time::Instant;

#[path = "../advancement.rs"] mod advancement;
#[path = "../anneal.rs"] mod anneal;
#[path = "../api.rs"] mod api;
#[path = "../bracket.rs"] mod bracket;
#[path = "../config.rs"] mod config;
#[path = "../elo.rs"] mod elo;
#[path = "../exact.rs"] mod exact;
#[path = "../ga.rs"] mod ga;
#[path = "../game_result.rs"] mod game_result;
#[path = "../ingest.rs"] mod ingest;
#[path = "../names.rs"] mod names;
#[path = "../optimize.rs"] mod optimize;
#[path = "../picks.rs"] mod picks;
#[path = "../pool.rs"] mod pool;
#[path = "../portfolio.rs"] mod portfolio;
#[path = "../score.rs"] mod score;
#[path = "../tree.rs"] mod tree;

use bracket::{Bracket, ScoringConfig};
use config::{Config, GaSettings};
use ga::{GeneticAlgorithm, MonteCarloScenarios, TeamRoundMutator, WholePortfolioGA};
use ingest::TournamentInfo;
use picks::Picks;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use score::ScenarioPool;
use std::time::Duration;

/// The 2023 men's field, from the shipped FiveThirtyEight forecast.
///
/// Benchmarking against invented ratings is a trap: a field with a wide rating
/// spread makes every game close to decided, which collapses the search space
/// and flatters any optimizer. The real spread is about 27 points across all 64
/// teams with heavy overlap between adjacent seeds, so upsets are common and
/// portfolio construction has something to actually do.
fn tournament() -> TournamentInfo {
    TournamentInfo::initialize("fivethirtyeight_ncaa_forecasts.csv")
        .expect("the shipped 2023 forecast should parse")
}

struct Bench {
    filter: Option<String>,
}

impl Bench {
    fn run<T>(&self, name: &str, iters: usize, mut f: impl FnMut() -> T) {
        if let Some(filter) = &self.filter {
            if !name.contains(filter.as_str()) {
                return;
            }
        }
        // Warm up so we are not measuring first-touch page faults.
        std::hint::black_box(f());
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(f());
        }
        let elapsed = start.elapsed();
        let per = elapsed.as_secs_f64() / iters as f64;
        println!(
            "{:<44} {:>12.3} us/iter   ({} iters, {:.2} s total)",
            name,
            per * 1e6,
            iters,
            elapsed.as_secs_f64()
        );
    }
}

fn ga_settings(population: usize, generations: usize) -> GaSettings {
    let mut s = GaSettings::default();
    s.population_size = population;
    s.generations = generations;
    s
}

fn main() {
    let filter = std::env::args().nth(1);
    let b = Bench { filter };

    let t = tournament();
    let scoring = ScoringConfig::default();
    let mut rng = SmallRng::seed_from_u64(0xBE_11_CE);

    // ---- Primitives -------------------------------------------------------
    let seed_picks = Picks::sample(&t, &mut rng);
    let other_picks = Picks::sample(&t, &mut rng);
    let seed_bracket = Bracket::from_picks(&t, &seed_picks, Some(&scoring));

    b.run("primitive/sample_picks", 200_000, || {
        Picks::sample(&t, &mut rng)
    });
    b.run("primitive/decode_bits", 500_000, || {
        Picks::from_bits(&t, seed_picks.bits())
    });
    b.run("primitive/force_to_round", 1_000_000, || {
        seed_picks.forced_to_round(&t, 17, 4)
    });
    b.run("primitive/build_display_bracket", 20_000, || {
        Bracket::from_picks(&t, &seed_picks, Some(&scoring))
    });
    b.run("primitive/disagreements", 500_000, || {
        seed_picks.disagreements(&other_picks)
    });

    // ---- Exact solver -----------------------------------------------------
    b.run("exact/solve", 20_000, || {
        exact::solve(&t, &scoring, &[]).unwrap().expected_value
    });

    // ---- Scenario pool ----------------------------------------------------
    b.run("pool/build_10k", 20, || {
        ScenarioPool::new(&t, 10_000, &scoring, 1)
    });

    let scen = ScenarioPool::new(&t, 10_000, &scoring, 1);
    let entries: Vec<Picks> = (0..5).map(|_| Picks::sample(&t, &mut rng)).collect();
    let prepared = scen.prepare_all(&entries);
    let candidate = scen.prepare(&seed_picks);
    let mut baseline = vec![0.0f32; scen.size()];
    scen.best_ball_into(&prepared, &mut baseline);

    b.run("score/single_vs_10k_serial", 5_000, || {
        scen.mean_score(&candidate)
    });
    b.run("score/single_vs_10k_parallel", 5_000, || {
        scen.par_mean_score(&candidate)
    });
    b.run("score/best_ball_5_vs_10k", 2_000, || {
        scen.best_ball_mean(&prepared)
    });
    b.run("score/marginal_vs_10k", 5_000, || {
        scen.mean_max_with(&candidate, &baseline)
    });

    let scenarios = MonteCarloScenarios::new(&t, 10_000, &scoring);
    b.run("score/api_single_vs_10k", 2_000, || {
        scenarios.score_bracket(&seed_bracket, &scoring)
    });

    // ---- End-to-end optimizers -------------------------------------------
    b.run("ga/single_100pop_20gen", 20, || {
        let mut ga = GeneticAlgorithm::with_seed(&t, ga_settings(100, 20), scoring, 7);
        ga.run(&t, &scenarios, false)
    });
    b.run("ga/whole_portfolio_5x_50pop_10gen", 10, || {
        let mut ga = WholePortfolioGA::with_seed(&t, 5, ga_settings(50, 10), scoring, 7);
        ga.run(&t, &scenarios, false)
    });

    let _ = Config::default();
    let _ = anneal::AnnealingConfig::default();
    let _ = TeamRoundMutator::random_move(&t, &mut rng);

    if b.filter.as_deref() == Some("quality") {
        quality_shootout(&t, &scoring);
    }
}

/// Head-to-head on the thing that actually matters: for a fixed budget, which
/// optimizer returns the portfolio with the higher best-ball payout — measured
/// on a *held-out* scenario pool, so fitting the training sample earns nothing.
fn quality_shootout(t: &TournamentInfo, scoring: &ScoringConfig) {
    const TRAIN: usize = 50_000;
    const HOLDOUT: usize = 200_000;

    let train = ScenarioPool::new(t, TRAIN, scoring, 1);
    let scenarios = MonteCarloScenarios::with_seed(t, TRAIN, scoring, 1);
    let locks = ga::LockSet::default();

    println!(
        "\n{:<26} {:>8} {:>12} {:>12} {:>10}",
        "portfolio strategy", "entries", "in-sample", "held-out", "seconds"
    );
    println!("{}", "-".repeat(72));

    for entries in [1usize, 3, 5, 10] {
        let mut rows: Vec<(String, f64, f64, Duration)> = Vec::new();

        let start = Instant::now();
        let plan = optimize::optimize(t, scoring, &train, &locks, entries, false);
        let elapsed = start.elapsed();
        let held = optimize::holdout_score(t, scoring, &plan.entries, HOLDOUT, 999);
        rows.push(("exact-basis".into(), plan.best_ball, held, elapsed));

        let start = Instant::now();
        let mut ga = WholePortfolioGA::with_seed(t, entries, ga_settings(100, 200), *scoring, 7);
        let brackets = ga.run(t, &scenarios, false);
        let elapsed = start.elapsed();
        let picks: Vec<Picks> = brackets.iter().map(|b| b.picks(t)).collect();
        let held = optimize::holdout_score(t, scoring, &picks, HOLDOUT, 999);
        let in_sample = train.par_best_ball_mean(&train.prepare_all(&picks));
        rows.push(("ga-whole".into(), in_sample, held, elapsed));

        let start = Instant::now();
        let seq = ga::SequentialPortfolioOptimizer::new(
            {
                let mut c = Config::default();
                c.ga = ga_settings(100, 200);
                c.simulation.pool_size = TRAIN;
                c
            },
            *scoring,
            locks.clone(),
        );
        let brackets = silently(|| seq.optimize(t, entries, false));
        let elapsed = start.elapsed();
        let picks: Vec<Picks> = brackets.iter().map(|b| b.picks(t)).collect();
        let held = optimize::holdout_score(t, scoring, &picks, HOLDOUT, 999);
        let in_sample = train.par_best_ball_mean(&train.prepare_all(&picks));
        rows.push(("ga-sequential".into(), in_sample, held, elapsed));

        for (name, in_sample, held, elapsed) in rows {
            println!(
                "{:<26} {:>8} {:>12.3} {:>12.3} {:>10.2}",
                name,
                entries,
                in_sample,
                held,
                elapsed.as_secs_f64()
            );
        }
        println!();
    }
}

/// The optimizers print progress; the shootout only wants their answers.
fn silently<T>(f: impl FnOnce() -> T) -> T {
    use std::io::Write;
    std::io::stdout().flush().ok();
    let saved = unsafe { libc_dup(1) };
    let devnull = std::fs::OpenOptions::new().write(true).open("/dev/null").unwrap();
    unsafe { libc_dup2(std_fd(&devnull), 1) };
    let out = f();
    std::io::stdout().flush().ok();
    unsafe {
        libc_dup2(saved, 1);
        libc_close(saved);
    }
    out
}

use std::os::unix::io::AsRawFd;
fn std_fd(f: &std::fs::File) -> i32 {
    f.as_raw_fd()
}
extern "C" {
    #[link_name = "dup"]
    fn libc_dup(fd: i32) -> i32;
    #[link_name = "dup2"]
    fn libc_dup2(old: i32, new: i32) -> i32;
    #[link_name = "close"]
    fn libc_close(fd: i32) -> i32;
}
