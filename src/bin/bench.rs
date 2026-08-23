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
#[path = "../field.rs"] mod field;
#[path = "../ga.rs"] mod ga;
#[path = "../game_result.rs"] mod game_result;
#[path = "../ingest.rs"] mod ingest;
#[path = "../ncaa_bracket.rs"] mod ncaa_bracket;
#[path = "../names.rs"] mod names;
#[path = "../optimize.rs"] mod optimize;
#[path = "../picks.rs"] mod picks;
#[path = "../pool.rs"] mod pool;
#[path = "../portfolio.rs"] mod portfolio;
#[path = "../score.rs"] mod score;
#[path = "../torvik.rs"] mod torvik;
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
    if b.filter.as_deref() == Some("hedge") {
        hedge_experiment(&t, &scoring);
    }
    if b.filter.as_deref() == Some("contrarian") {
        contrarian_experiment(&t, &scoring);
    }
    if b.filter.as_deref() == Some("sweep") {
        entry_count_sweep(&t, &scoring);
    }
}

/// What does each additional entry buy you?
///
/// Holds the opposing field at 100 entries and sweeps how many brackets you
/// submit, reporting how often one of them finishes first. Emits CSV on stdout.
///
/// Every number is measured **out of sample**: portfolios are built against one
/// pool of tournaments and one set of field draws, then scored against an
/// independently seeded pool and an independently drawn field. Reporting the
/// training score instead would flatter the curve, because the search picked
/// those entries to beat those particular sampled opponents.
fn entry_count_sweep(t: &TournamentInfo, scoring: &ScoringConfig) {
    const TRAIN_SCENARIOS: usize = 30_000;
    const TEST_SCENARIOS: usize = 120_000;
    const REPLICATES: usize = 6;
    const TEST_REPLICATES: usize = 12;
    const OPPONENTS: usize = 100;
    const MAX_ENTRIES: usize = 16;

    let train = ScenarioPool::new(t, TRAIN_SCENARIOS, scoring, 1);
    let test = ScenarioPool::new(t, TEST_SCENARIOS, scoring, 0xDEC0DE);
    let locks = ga::LockSet::default();

    let public = match field::fetch_espn(2023, "./data")
        .and_then(|f| field::PickPopularity::from_file(&f, t))
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("no ESPN pick data ({}); using the chalk fallback", e);
            field::PickPopularity::chalk(t, 1.6)
        }
    };
    let public = &public;

    let draw = |salt: u64| {
        move |replicate: usize, i: usize| {
            let mut rng = SmallRng::seed_from_u64(
                salt ^ ((replicate as u64) << 40)
                    ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            public.sample_entry(t, &mut rng)
        }
    };

    // The competition does not change as your entry count does, so each field
    // is sampled and reduced to per-scenario order statistics once.
    let training_field = train.competition(REPLICATES, OPPONENTS, draw(0x5EED));
    let holdout_field = test.competition(TEST_REPLICATES, OPPONENTS, draw(0xA11CE));

    eprintln!(
        "field: {}, {} opposing entries; train {} scenarios x {} draws; test {} x {}",
        public.source, OPPONENTS, TRAIN_SCENARIOS, REPLICATES, TEST_SCENARIOS, TEST_REPLICATES
    );

    println!("entries,total_pool,win_optimized,points_optimized,fair_share,in_sample,champions");
    let mut profile = vec![0.0f32; test.size()];

    for entries in 1..=MAX_ENTRIES {
        let fp = optimize::optimize_for(
            t,
            scoring,
            &train,
            &locks,
            entries,
            optimize::Objective::FirstPlace(&training_field),
            false,
        );

        // The same number of entries, chosen to maximize points instead.
        let bb = optimize::optimize_for(
            t, scoring, &train, &locks, entries, optimize::Objective::BestBall, false,
        );

        test.best_ball_into(&test.prepare_all(&fp.entries), &mut profile);
        let win_held = test.mean_win_share(&profile, &holdout_field);
        test.best_ball_into(&test.prepare_all(&bb.entries), &mut profile);
        let points_held = test.mean_win_share(&profile, &holdout_field);

        let total = OPPONENTS + entries;
        let mut champions: Vec<&str> = fp
            .entries
            .iter()
            .map(|p| t.teams[p.champion() as usize].name.as_str())
            .collect();
        champions.sort_unstable();
        champions.dedup();

        println!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{}",
            entries,
            total,
            win_held,
            points_held,
            entries as f64 / total as f64,
            fp.best_ball,
            champions.join(" ")
        );
    }
}

/// How should pool size change what you enter?
///
/// Scores three portfolios by how often they finish first against the real 2023
/// ESPN public field, across pool sizes: the exact expected-score optimum, a
/// best-ball-optimized portfolio, and one optimized for first place directly.
fn contrarian_experiment(t: &TournamentInfo, scoring: &ScoringConfig) {
    const SCENARIOS: usize = 40_000;
    const REPLICATES: usize = 8;

    let pool = ScenarioPool::new(t, SCENARIOS, scoring, 1);
    let locks = ga::LockSet::default();

    let public = match field::fetch_espn(2023, "./data")
        .and_then(|f| field::PickPopularity::from_file(&f, t))
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("no ESPN pick data ({}); using the chalk fallback", e);
            field::PickPopularity::chalk(t, 1.6)
        }
    };
    let (fav, share) = public.favourite();
    println!(
        "\nPublic field: {} — most-picked champion {} at {:.1}%",
        public.source,
        t.teams[fav as usize].name,
        share * 100.0
    );

    let exact_best = exact::solve(t, scoring, &[]).unwrap();
    let ev_picks = Picks::from_winners(t, &exact_best.bracket.winner_indices());

    println!("\nP(finish first), by what the portfolio was built to maximize:");
    println!(
        "\n{:>6}  {:>7}  {:>12}  {:>12}  {:>12}  {:>7}",
        "pool", "entries", "EV-optimal", "best-ball", "first-place", "vs EV"
    );
    println!("{}", "-".repeat(70));

    for (entrants, entries) in [(20usize, 1usize), (100, 1), (100, 3), (1_000, 3), (10_000, 3)] {
        let opponents = entrants.saturating_sub(entries).max(1);
        let competition = pool.competition(REPLICATES, opponents, |replicate, i| {
            let mut rng = SmallRng::seed_from_u64(
                0xF1E1D_u64 ^ ((replicate as u64) << 40) ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            public.sample_entry(t, &mut rng)
        });

        // The EV-optimal bracket, repeated to fill the entry slots it is
        // allowed — the naive multi-entry version of "just play the best one".
        let ev_entries = vec![ev_picks; entries];
        let mut profile = vec![0.0f32; pool.size()];
        pool.best_ball_into(&pool.prepare_all(&ev_entries), &mut profile);
        let ev_win = pool.mean_win_share(&profile, &competition);

        let bb = optimize::optimize_for(
            t, scoring, &pool, &locks, entries, optimize::Objective::BestBall, false,
        );
        pool.best_ball_into(&pool.prepare_all(&bb.entries), &mut profile);
        let bb_win = pool.mean_win_share(&profile, &competition);

        let fp = optimize::optimize_for(
            t,
            scoring,
            &pool,
            &locks,
            entries,
            optimize::Objective::FirstPlace(&competition),
            false,
        );

        println!(
            "{:>6}  {:>7}  {:>11.3}%  {:>11.3}%  {:>11.3}%   {:>5.2}x",
            entrants,
            entries,
            ev_win * 100.0,
            bb_win * 100.0,
            fp.best_ball * 100.0,
            fp.best_ball / ev_win,
        );

        let champions: Vec<String> = fp
            .entries
            .iter()
            .map(|p| {
                let b = Bracket::from_picks(t, p, Some(scoring));
                format!(
                    "{} (EV {:+.0})",
                    b.winner.name,
                    exact::expected_value(t, &b, scoring) - exact_best.expected_value
                )
            })
            .collect();
        println!("          first-place entries: {}", champions.join(", "));
    }
}

/// Does the EV-optimal bracket belong in a multi-entry portfolio?
///
/// Compares two ways of building one: pin the exact single-bracket optimum as
/// entry 1 and hedge around it, versus letting every entry move freely.
fn hedge_experiment(t: &TournamentInfo, scoring: &ScoringConfig) {
    const TRAIN: usize = 50_000;
    const HOLDOUT: usize = 200_000;

    let train = ScenarioPool::new(t, TRAIN, scoring, 1);
    let locks = ga::LockSet::default();
    let exact_best = exact::solve(t, scoring, &[]).unwrap();
    let optimum = Picks::from_winners(t, &exact_best.bracket.winner_indices());

    println!(
        "\nExact single-bracket optimum: EV {:.2}, champion {}",
        exact_best.expected_value, exact_best.bracket.winner.name
    );
    println!(
        "\n{:<8} {:>14} {:>14} {:>10} {:>26}",
        "entries", "pinned", "free", "free gain", "optimum still an entry?"
    );
    println!("{}", "-".repeat(78));

    for k in [2usize, 3, 5, 10] {
        let free = optimize::optimize_for(t, scoring, &train, &locks, k, optimize::Objective::BestBall, false);
        let pinned = pinned_portfolio(t, scoring, &train, &locks, k, optimum);

        let free_held = optimize::holdout_score(t, scoring, &free.entries, HOLDOUT, 999);
        let pinned_held = optimize::holdout_score(t, scoring, &pinned, HOLDOUT, 999);
        let kept = free.entries.iter().any(|e| e.bits() == optimum.bits());

        println!(
            "{:<8} {:>14.3} {:>14.3} {:>+10.3} {:>26}",
            k,
            pinned_held,
            free_held,
            free_held - pinned_held,
            if kept { "yes" } else { "no" }
        );

        if k == 5 {
            println!("\n  the five free entries, against the exact optimum:");
            println!(
                "  {:<4} {:<16} {:>10} {:>12} {:>14}",
                "#", "champion", "solo EV", "EV given up", "games differing"
            );
            for (i, e) in free.entries.iter().enumerate() {
                let b = Bracket::from_picks(t, e, Some(scoring));
                let ev = exact::expected_value(t, &b, scoring);
                println!(
                    "  {:<4} {:<16} {:>10.2} {:>12.2} {:>14}",
                    i + 1,
                    b.winner.name,
                    ev,
                    ev - exact_best.expected_value,
                    e.disagreements(&optimum)
                );
            }
            println!();
        }
    }
}

/// Greedy selection plus coordinate ascent, with slot 0 frozen to `pinned`.
fn pinned_portfolio(
    t: &TournamentInfo,
    scoring: &ScoringConfig,
    pool: &ScenarioPool,
    locks: &ga::LockSet,
    entries: usize,
    pinned: Picks,
) -> Vec<Picks> {
    let basis = optimize::exact_basis(t, scoring, locks);
    let prepared: Vec<_> = basis.iter().map(|p| pool.prepare(p)).collect();

    let mut portfolio = vec![pinned];
    let mut baseline = vec![0.0f32; pool.size()];
    pool.absorb_into(&pool.prepare(&pinned), &mut baseline);

    while portfolio.len() < entries {
        let best = prepared
            .iter()
            .enumerate()
            .map(|(i, c)| (i, pool.mean_max_with(c, &baseline)))
            .fold((0usize, f64::NEG_INFINITY), |b, n| if n.1 > b.1 { n } else { b });
        pool.absorb_into(&prepared[best.0], &mut baseline);
        portfolio.push(basis[best.0]);
    }

    // Coordinate ascent over slots 1.. only; slot 0 never moves.
    let mut current = pool.par_best_ball_mean(&pool.prepare_all(&portfolio));
    for _ in 0..32 {
        let mut improved = false;
        for slot in 1..portfolio.len() {
            let others: Vec<_> = portfolio
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != slot)
                .map(|(_, p)| pool.prepare(p))
                .collect();
            let mut base = vec![0.0f32; pool.size()];
            pool.best_ball_into(&others, &mut base);

            let incumbent = portfolio[slot];
            let mut candidates: Vec<Picks> = basis.clone();
            for (team, wins) in picks::all_moves() {
                candidates.push(incumbent.forced_to_round(t, team, wins));
            }
            let best = candidates
                .iter()
                .enumerate()
                .map(|(i, p)| (i, pool.mean_max_with(&pool.prepare(p), &base)))
                .fold((0usize, f64::NEG_INFINITY), |b, n| if n.1 > b.1 { n } else { b });

            if best.1 > current + 1e-9 {
                portfolio[slot] = candidates[best.0];
                current = best.1;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    portfolio
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
        let plan = optimize::optimize_for(t, scoring, &train, &locks, entries, optimize::Objective::BestBall, false);
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
