//! Opponent-adjusted team ratings, and the scale that turns them into
//! probabilities.
//!
//! Elo learns strength of schedule transitively, one game at a time. With ~360
//! teams playing ~30 games each on near-disjoint schedules it never finishes
//! propagating, and a 15-0 mid-major stays overrated into March. This module
//! solves for every team at once instead: one ridge least-squares fit of
//!
//! ```text
//! margin(game) = rating[home] - rating[away] + home_edge * (not neutral)
//! ```
//!
//! over the season's game log. Measured over nine seasons and 600 NCAA
//! tournament games (`analysis/win-probability/`, `docs/WIN_PROBABILITY_METHODS.md`),
//! that is 0.065 of log loss and eight points of tournament accuracy better than
//! the Elo path, and the bootstrap interval on the gap excludes zero by a wide
//! margin. It is also within noise of BartTorvik's published T-Rank, which is
//! the check that the implementation is honest rather than merely different.
//!
//! Ratings are in **points of scoring margin against an average Division I
//! team**. Nothing here needs possession counts, tempo, or box scores — only the
//! date, the two teams, the site, and the score, which is a strict subset of
//! what `torvik.rs` already parses. Splitting each team into offence and defence
//! per possession scored 0.0008 better, which is not a difference, and would
//! have tied the model to one feed's extra columns.

use crate::elo::ELO_PER_538_POINT;
use crate::game_result::GameResult;
use crate::names;
use std::collections::BTreeMap;

/// Points of scoring margin per unit of logit: `P(win) = logistic(margin / 7.109)`.
///
/// Fitted over 32,571 post-November games with the calibration held out a season
/// at a time. The equivalent probit fit (σ = 11.9 points) scored 0.0002 better,
/// which is nothing — either link is right, and this one composes with the
/// existing 538 scale for free.
pub const POINTS_PER_LOGIT: f64 = 7.109;

/// Home-court edge in points, used only when the fit cannot estimate one.
///
/// The per-season fitted values over 2017-2026 were 3.01, 3.54, 3.21, 2.44,
/// 2.79, 3.54, 3.11, 3.24 and 2.92 — mean 3.09, drifting down. The Elo path's
/// `HOME_ADVANTAGE = 100.0` is 4.09 points on its own scale, about 30% hot.
pub const DEFAULT_HOME_EDGE: f64 = 3.09;

/// Ridge penalty on the team columns.
///
/// The result is flat between 0.5 and 2 and starts to hurt above 8: at λ = 32
/// the fit is worse than Elo, because over-shrinking compresses exactly the
/// strong teams a tournament field is made of.
pub const RIDGE_LAMBDA: f64 = 1.0;

/// The 538 rating of an average Division I team. Anchored to Elo's own anchor so
/// the two rating sources land on the same scale and can be raced directly.
pub const AVERAGE_RATING_538: f64 = crate::elo::SCALE_ANCHOR_538;

/// 538 rating points per point of expected scoring margin.
///
/// `ProbabilityCache` computes `1 / (1 + 10^(-diff * 30.464 / 400))`, which is
/// `logistic(diff * 30.464 * ln 10 / 400)`. Setting that exponent equal to
/// `margin / POINTS_PER_LOGIT` is what makes the fitted link come out of the
/// cache unchanged — so adopting this model costs `ProbabilityCache` nothing,
/// and the 64x64 table the optimizers actually read is byte-for-byte the same
/// shape it always was.
pub const RATING_538_PER_POINT: f64 =
    400.0 / (ELO_PER_538_POINT * std::f64::consts::LN_10) / POINTS_PER_LOGIT;

/// Strength of each seed in points, seeds 1 through 16.
///
/// The backup when no game log can be had. Fitted by penalized logistic
/// regression on the same 600 tournament games, then projected onto the
/// monotone (non-increasing) cone — the unconstrained fit rates 9-seeds above
/// 8-seeds and 15-seeds above 14-seeds, which is 600 games of noise and would be
/// indefensible in a fallback. Monotone costs 0.001 of log loss against the
/// unconstrained fit and still beats a linear-in-seed-difference model by 0.009.
///
/// On its own this scores 0.562 on tournament games: worse than the ridge fit
/// (0.544) but comfortably better than the Elo path it replaces (0.608). A run
/// that falls back to it is degraded, not broken.
pub const SEED_POINTS: [f64; 16] = [
    12.9708, 5.3774, 4.5463, 4.3961, 3.4304, 1.4429, 0.7621, 0.7621, 0.7621, -1.0443, -1.0443,
    -2.4553, -6.1923, -7.9046, -7.9046, -7.9046,
];

/// Convert a rating in points of margin to the 538 scale `ProbabilityCache` reads.
#[inline]
pub fn points_to_538(points: f64) -> f32 {
    (AVERAGE_RATING_538 + points * RATING_538_PER_POINT) as f32
}

/// The rating of a team known only by its seed.
pub fn seed_rating_538(seed: i32) -> Result<f32, String> {
    if !(1..=16).contains(&seed) {
        return Err(format!("seed {} is outside 1-16", seed));
    }
    Ok(points_to_538(SEED_POINTS[(seed - 1) as usize]))
}

/// Ratings for every team that appeared in a season's game log.
#[derive(Debug, Clone)]
pub struct RidgeRatings {
    /// Display names, sorted, parallel to `points`.
    ///
    /// Held as a prepared pair rather than rebuilt per lookup: resolving all 64
    /// bracket entries against a `HashMap` of ~360 teams otherwise clones and
    /// re-sorts the candidate list 64 times over.
    names: Vec<String>,
    points: Vec<f64>,
    /// Fitted home-court edge, in points.
    pub home_edge: f64,
    pub games_used: usize,
}

impl RidgeRatings {
    pub fn team_count(&self) -> usize {
        self.names.len()
    }

    /// Points above an average Division I team, by team id.
    pub fn points_for_name(&self, name: &str) -> Result<f64, names::NameError> {
        names::resolve(name, &self.names).map(|i| self.points[i])
    }

    /// The same rating on the 538 scale.
    pub fn rating_538_for_name(&self, name: &str) -> Result<f32, names::NameError> {
        self.points_for_name(name).map(points_to_538)
    }

    /// The strongest `n` teams, for reporting.
    pub fn top_teams(&self, n: usize) -> Vec<(&str, f64)> {
        let mut idx: Vec<usize> = (0..self.points.len()).collect();
        // Ties broken by name so repeated runs print the same order.
        idx.sort_by(|&a, &b| {
            self.points[b]
                .partial_cmp(&self.points[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.names[a].cmp(&self.names[b]))
        });
        idx.into_iter()
            .take(n)
            .map(|i| (self.names[i].as_str(), self.points[i]))
            .collect()
    }

    pub fn print_top_teams(&self, n: usize) {
        println!("\nTop {} Teams by adjusted rating:", n);
        println!("{:>4} {:>8}  {}", "Rank", "Points", "Team");
        println!("{}", "-".repeat(44));
        for (i, (name, points)) in self.top_teams(n).iter().enumerate() {
            println!(
                "{:>4} {:>+8.2}  {}",
                i + 1,
                points,
                &name[..name.len().min(28)]
            );
        }
        println!(
            "\n(points of scoring margin against an average D-I team; \
             fitted home edge {:+.2})",
            self.home_edge
        );
    }
}

/// Fit ratings to a season's games.
///
/// One pass to accumulate the normal equations and one Cholesky solve. The
/// design matrix is never formed: each game touches exactly three columns, so
/// accumulation is O(games), not O(games x teams).
pub fn fit(games: &[GameResult], lambda: f64) -> Result<RidgeRatings, String> {
    // Collected into a BTreeMap and *then* numbered, rather than numbered on
    // first appearance: column order decides the pivot order of the Cholesky, so
    // numbering by sorted id makes the fit invariant to the order the caller
    // hands the games over in, not merely repeatable for one fixed order.
    let mut by_id: BTreeMap<&str, &str> = BTreeMap::new();
    let mut played = 0usize;
    for game in games.iter().filter(|g| g.is_completed) {
        by_id.entry(&game.home_team_id).or_insert(&game.home_team_name);
        by_id.entry(&game.away_team_id).or_insert(&game.away_team_name);
        played += 1;
    }

    let n = by_id.len();
    if n < 2 || played == 0 {
        return Err(format!(
            "not enough games to fit ratings: {} completed game(s) over {} team(s)",
            played, n
        ));
    }

    let mut index: BTreeMap<&str, usize> = BTreeMap::new();
    let mut display: Vec<&str> = Vec::with_capacity(n);
    for (i, (id, name)) in by_id.iter().enumerate() {
        index.insert(id, i);
        display.push(name);
    }

    let p = n + 1;
    let home = n;
    let mut ata = vec![0.0f64; p * p];
    let mut aty = vec![0.0f64; p];

    for game in games.iter().filter(|g| g.is_completed) {
        let a = index[game.home_team_id.as_str()];
        let b = index[game.away_team_id.as_str()];
        if a == b {
            continue;
        }
        let h = if game.is_neutral_site { 0.0 } else { 1.0 };
        let margin = game.home_score as f64 - game.away_score as f64;

        ata[a * p + a] += 1.0;
        ata[b * p + b] += 1.0;
        ata[a * p + b] -= 1.0;
        ata[b * p + a] -= 1.0;
        if h != 0.0 {
            ata[a * p + home] += h;
            ata[home * p + a] += h;
            ata[b * p + home] -= h;
            ata[home * p + b] -= h;
            ata[home * p + home] += h * h;
        }
        aty[a] += margin;
        aty[b] -= margin;
        aty[home] += h * margin;
    }

    // The team columns are translation-invariant — adding a constant to every
    // rating changes no margin — so without a penalty the system is singular.
    // The ridge term both fixes that and does the early-season shrinking.
    for t in 0..n {
        ata[t * p + t] += lambda;
    }
    // The home column needs no shrinking, only enough to stay positive definite
    // before any non-neutral game has been seen.
    ata[home * p + home] += 1e-6;

    let solution = cholesky_solve(ata, aty, p)?;

    let mut points: Vec<f64> = solution[..n].to_vec();
    // Ridge already lands near the minimum-norm (mean-zero) solution; centring
    // makes "points above average" exact rather than approximate.
    let mean = points.iter().sum::<f64>() / n as f64;
    for v in points.iter_mut() {
        *v -= mean;
    }

    Ok(RidgeRatings {
        names: display.into_iter().map(str::to_string).collect(),
        points,
        home_edge: solution[home],
        games_used: played,
    })
}

/// Dot product with the reduction split four ways.
///
/// Spelled as a single accumulator, this is a serial dependency chain — one
/// multiply-add per iteration at the latency of an FMA, and LLVM will not
/// reassociate floating point on its own. Four independent accumulators let the
/// pipeline stay full and the loop vectorize, and the summation order is still
/// fixed, so the fit stays bit-for-bit reproducible run to run.
#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let chunks = n / 4;
    let mut acc = [0.0f64; 4];
    for c in 0..chunks {
        let i = c * 4;
        acc[0] += x[i] * y[i];
        acc[1] += x[i + 1] * y[i + 1];
        acc[2] += x[i + 2] * y[i + 2];
        acc[3] += x[i + 3] * y[i + 3];
    }
    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for i in (chunks * 4)..n {
        total += x[i] * y[i];
    }
    total
}

/// Solve `A x = b` for symmetric positive-definite `A`, in place.
///
/// A failed factorization is returned as an error rather than papered over: a
/// non-positive pivot means the fit is degenerate, and the caller has a seed
/// fallback that is honest about being one.
fn cholesky_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>, String> {
    for j in 0..n {
        let diag = a[j * n + j] - dot(&a[j * n..j * n + j], &a[j * n..j * n + j]);
        if !(diag > 0.0) {
            return Err(format!(
                "rating fit is degenerate: non-positive pivot {} at column {} of {}",
                diag, j, n
            ));
        }
        let l = diag.sqrt();
        a[j * n + j] = l;
        // Split so the pivot row can be borrowed immutably while the rows below
        // it are written. Both slices are contiguous, which is what lets the dot
        // product below vectorize.
        let (head, tail) = a.split_at_mut((j + 1) * n);
        let pivot = &head[j * n..j * n + j];
        for (i, row) in tail.chunks_exact_mut(n).enumerate() {
            let s = row[j] - dot(&row[..j], pivot);
            row[j] = s / l;
            let _ = i;
        }
    }

    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn game(day: u32, home: &str, away: &str, hs: u32, aws: u32, neutral: bool) -> GameResult {
        GameResult {
            game_id: format!("{}-{}-{}", day, home, away),
            date: NaiveDate::from_ymd_opt(2025, 11, day).unwrap(),
            home_team_id: names::normalize(home),
            home_team_name: home.to_string(),
            away_team_id: names::normalize(away),
            away_team_name: away.to_string(),
            home_score: hs,
            away_score: aws,
            is_neutral_site: neutral,
            is_conference_game: false,
            is_completed: true,
        }
    }

    /// A round robin with a known ordering has to come out in that order, and
    /// the ratings have to be centred on zero.
    #[test]
    fn a_round_robin_recovers_the_ordering() {
        // A beats B by 10, B beats C by 10, A beats C by 20 — all neutral, so
        // there is an exact fit with A=+10, B=0, C=-10 before shrinking.
        let games = vec![
            game(1, "Alpha", "Bravo", 80, 70, true),
            game(2, "Bravo", "Charlie", 80, 70, true),
            game(3, "Alpha", "Charlie", 90, 70, true),
        ];
        let r = fit(&games, RIDGE_LAMBDA).expect("fits");
        let a = r.points_for_name("Alpha").unwrap();
        let b = r.points_for_name("Bravo").unwrap();
        let c = r.points_for_name("Charlie").unwrap();
        assert!(a > b && b > c, "{} {} {}", a, b, c);
        assert!((a + b + c).abs() < 1e-9, "ratings are centred: {}", a + b + c);
        assert!(b.abs() < 1e-9, "the middle team is average: {}", b);
    }

    /// The whole point of the model: schedule strength is solved for, not
    /// accumulated. A team that only ever played the worst team in the league
    /// must not outrate a team that only ever played the best.
    #[test]
    fn strength_of_schedule_is_solved_not_accumulated() {
        let mut games = vec![
            // Strong and Weak establish the ends of the league.
            game(1, "Strong", "Weak", 100, 60, true),
            game(2, "Strong", "Weak", 100, 60, true),
            game(3, "Strong", "Weak", 100, 60, true),
        ];
        // Padder beats only Weak, by 12. Tester loses to Strong by 12.
        for d in 4..12 {
            games.push(game(d, "Padder", "Weak", 82, 70, true));
            games.push(game(d, "Strong", "Tester", 82, 70, true));
        }
        let r = fit(&games, RIDGE_LAMBDA).expect("fits");
        let padder = r.points_for_name("Padder").unwrap();
        let tester = r.points_for_name("Tester").unwrap();
        assert!(
            tester > padder,
            "an 0-8 team against the best should outrate an 8-0 team against the worst: \
             Tester {:.2}, Padder {:.2}",
            tester,
            padder
        );
    }

    /// The home column has to absorb the home edge rather than leaking it into
    /// the team ratings.
    #[test]
    fn the_home_edge_is_estimated_not_absorbed() {
        // Two identical teams, alternating hosts, home team always wins by 6.
        let mut games = Vec::new();
        for d in 1..=20 {
            if d % 2 == 0 {
                games.push(game(d, "Alpha", "Bravo", 76, 70, false));
            } else {
                games.push(game(d, "Bravo", "Alpha", 76, 70, false));
            }
        }
        let r = fit(&games, RIDGE_LAMBDA).expect("fits");
        assert!(
            (r.home_edge - 6.0).abs() < 0.5,
            "home edge should be ~6, got {}",
            r.home_edge
        );
        let a = r.points_for_name("Alpha").unwrap();
        let b = r.points_for_name("Bravo").unwrap();
        assert!(
            (a - b).abs() < 0.5,
            "evenly matched teams should stay even: {} vs {}",
            a,
            b
        );
    }

    /// Neutral games carry no home term, which is what makes the tournament
    /// prediction a pure rating difference.
    #[test]
    fn neutral_games_do_not_feed_the_home_column() {
        let games: Vec<GameResult> = (1..=10)
            .map(|d| game(d, "Alpha", "Bravo", 80, 70, true))
            .collect();
        let r = fit(&games, RIDGE_LAMBDA).expect("fits");
        assert!(
            r.home_edge.abs() < 1e-6,
            "no non-neutral games, so no home edge: {}",
            r.home_edge
        );
    }

    /// The conversion exists to leave `ProbabilityCache` alone: a rating
    /// difference of `m` points, pushed through the cache's own formula, has to
    /// come back out as `logistic(m / POINTS_PER_LOGIT)`.
    #[test]
    fn the_538_conversion_reproduces_the_fitted_link() {
        for margin in [-25.0, -7.0, 0.0, 1.5, 12.0, 30.0] {
            let a = points_to_538(margin) as f64;
            let b = points_to_538(0.0) as f64;
            let from_cache = 1.0 / (1.0 + 10f64.powf(-(a - b) * ELO_PER_538_POINT / 400.0));
            let from_link = 1.0 / (1.0 + (-margin / POINTS_PER_LOGIT).exp());
            assert!(
                (from_cache - from_link).abs() < 1e-4,
                "margin {}: cache {} vs link {}",
                margin,
                from_cache,
                from_link
            );
        }
    }

    /// The seed table is the fallback, so its own sanity is worth a test: it has
    /// to be monotone, and it has to produce probabilities in the range the
    /// tournament actually shows.
    #[test]
    fn the_seed_table_is_monotone_and_sanely_scaled() {
        for i in 1..16 {
            assert!(
                SEED_POINTS[i - 1] >= SEED_POINTS[i],
                "seed {} rated below seed {}",
                i,
                i + 1
            );
        }
        let one = seed_rating_538(1).unwrap() as f64;
        let sixteen = seed_rating_538(16).unwrap() as f64;
        let eight = seed_rating_538(8).unwrap() as f64;
        let nine = seed_rating_538(9).unwrap() as f64;
        let p = |a: f64, b: f64| 1.0 / (1.0 + 10f64.powf(-(a - b) * ELO_PER_538_POINT / 400.0));
        assert!(
            p(one, sixteen) > 0.90 && p(one, sixteen) < 0.995,
            "1 vs 16: {}",
            p(one, sixteen)
        );
        assert!(
            (p(eight, nine) - 0.5).abs() < 0.02,
            "8 vs 9 is a coin flip: {}",
            p(eight, nine)
        );
        assert!(seed_rating_538(0).is_err() && seed_rating_538(17).is_err());
    }

    /// An empty or single-team log is an error, not a field of average teams.
    #[test]
    fn too_little_data_is_an_error() {
        assert!(fit(&[], RIDGE_LAMBDA).is_err());
        assert!(fit(&[game(1, "Alpha", "Alpha", 80, 70, true)], RIDGE_LAMBDA).is_err());
    }

    /// Ratings decide which bracket the optimizer returns, so the same log has
    /// to give the same ratings on every run.
    #[test]
    fn the_fit_is_deterministic() {
        let games: Vec<GameResult> = (1..=25)
            .map(|d| {
                let (h, a) = match d % 3 {
                    0 => ("Alpha", "Bravo"),
                    1 => ("Bravo", "Charlie"),
                    _ => ("Charlie", "Alpha"),
                };
                game(d, h, a, 70 + d, 70, d % 2 == 0)
            })
            .collect();
        let first = fit(&games, RIDGE_LAMBDA).expect("fits");
        for _ in 0..5 {
            let again = fit(&games, RIDGE_LAMBDA).expect("fits");
            assert_eq!(first.names, again.names);
            assert_eq!(first.points, again.points);
            assert_eq!(first.home_edge, again.home_edge);
        }
    }
}
