//! Portfolio construction for pools that allow more than one entry.
//!
//! With one entry the problem is solved: expected score is linear in the picks,
//! so [`crate::exact`] returns the provably optimal bracket in microseconds.
//! With `k` entries and best-ball payout — only your best bracket counts — the
//! objective becomes `E_s[max_i score(b_i, s)]`, a maximum inside an
//! expectation, with no closed form and no linearity to exploit.
//!
//! The genetic algorithms search that space from random brackets. This module
//! does something strictly stronger, in less time, by using two facts the GA
//! throws away.
//!
//! **A good portfolio is made of conditionally optimal brackets.** The reason to
//! enter a second bracket is to cover an outcome the first one misses, and the
//! best way to cover "Duke wins it all" is the highest-expected-value bracket
//! *given* Duke wins it all — which the exact solver produces by adding one
//! lock. Sweeping every team against every depth gives a few hundred such
//! brackets for a couple of milliseconds of work, and every one of them is
//! optimal for the bet it represents. That is the basis the search starts from,
//! instead of noise.
//!
//! **Holding the rest of the portfolio fixed makes one entry's problem easy.**
//! If entries other than `i` are frozen, their per-scenario best is a constant
//! vector, and the fitness of any replacement for `i` is one bracket's worth of
//! scoring against that baseline. So the search sweeps entries one at a time,
//! trying every conditionally optimal bracket and every team-round move away
//! from the incumbent, and keeps the best. Each accepted swap raises best-ball
//! by construction, so the sweep converges — this is coordinate ascent, and it
//! terminates at a portfolio no single-entry change can improve.
//!
//! What it does not claim: this is a strong local optimum, not the global one.
//! Nothing tractable gives the global optimum for best-ball. But it dominates a
//! random-initialized GA on both axes that matter — it scores higher, and it is
//! deterministic, so two runs give the same answer.

use crate::bracket::{Bracket, ScoringConfig};
use crate::exact::{self, TeamLock};
use crate::ga::LockSet;
use crate::ingest::TournamentInfo;
use crate::picks::{all_moves, Picks};
use crate::score::{Competition, ScenarioPool, ScoredPicks};
use rayon::prelude::*;
use std::collections::HashSet;

/// What the portfolio search maximizes.
///
/// Both variants reduce a portfolio to one number per scenario — the best score
/// any of its entries posts — and then differ only in what they do with it. That
/// is what lets one search serve both: greedy selection and coordinate ascent
/// never see the objective, only "score this candidate against this baseline".
#[derive(Clone, Copy)]
pub enum Objective<'a> {
    /// Expected best-ball score. Correct when the payout is linear in points.
    BestBall,
    /// Expected share of first place against a sampled public field. Correct
    /// when the pool pays the top finisher, and the only one of the two that
    /// prices in being picked by everyone else.
    FirstPlace(&'a Competition),
}

impl<'a> Objective<'a> {
    /// Value of adding `candidate` to a portfolio whose per-scenario best is
    /// `baseline`.
    #[inline]
    fn value(&self, pool: &ScenarioPool, candidate: &ScoredPicks, baseline: &[f32]) -> f64 {
        match self {
            Objective::BestBall => pool.mean_max_with(candidate, baseline),
            Objective::FirstPlace(c) => pool.mean_win_share_with(candidate, baseline, c),
        }
    }

    /// Value of a portfolio already reduced to its per-scenario best.
    fn value_of_profile(&self, pool: &ScenarioPool, profile: &[f32]) -> f64 {
        match self {
            Objective::BestBall => {
                profile.iter().map(|&v| v as f64).sum::<f64>() / profile.len() as f64
            }
            Objective::FirstPlace(c) => pool.mean_win_share(profile, c),
        }
    }

    /// Value of a whole portfolio.
    fn value_of(&self, pool: &ScenarioPool, entries: &[ScoredPicks]) -> f64 {
        let mut profile = vec![0.0f32; pool.size()];
        pool.best_ball_into(entries, &mut profile);
        self.value_of_profile(pool, &profile)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Objective::BestBall => "best-ball score",
            Objective::FirstPlace(_) => "P(finish first)",
        }
    }
}

/// A finished portfolio and how it was arrived at.
pub struct PortfolioPlan {
    pub entries: Vec<Picks>,
    /// Best-ball score against the pool it was optimized on.
    pub best_ball: f64,
    /// Best-ball score of the first entry alone, for reference.
    pub single_entry: f64,
    /// Conditionally optimal brackets the search could draw on.
    pub basis_size: usize,
    /// Improving sweeps the local search ran before it converged.
    pub sweeps: usize,
    /// What the local search added on top of the greedy selection. Never
    /// negative: every accepted swap raises the objective by construction.
    pub polish_gain: f64,
}

impl PortfolioPlan {
    /// Average number of games on which two entries disagree.
    ///
    /// A portfolio of near-identical brackets scores barely more than one of
    /// them; this is the cheapest sanity check that the entries are actually
    /// covering different outcomes.
    pub fn mean_spread(&self) -> f64 {
        if self.entries.len() < 2 {
            return 0.0;
        }
        let mut total = 0usize;
        let mut pairs = 0usize;
        for i in 0..self.entries.len() {
            for j in i + 1..self.entries.len() {
                total += self.entries[i].disagreements(&self.entries[j]);
                pairs += 1;
            }
        }
        total as f64 / pairs as f64
    }

    pub fn brackets(
        &self,
        tournament: &TournamentInfo,
        scoring: &ScoringConfig,
    ) -> Vec<Bracket> {
        self.entries
            .iter()
            .map(|p| Bracket::from_picks(tournament, p, Some(scoring)))
            .collect()
    }
}

/// Every bracket that is exactly optimal subject to one extra advancement
/// requirement, plus the unconstrained optimum.
///
/// One solve is a few microseconds, so the whole 64-team by 6-round sweep costs
/// less than a single Monte Carlo evaluation. Requirements that conflict with
/// the user's own locks are skipped rather than reported: they are simply
/// brackets that cannot exist.
pub fn exact_basis(
    tournament: &TournamentInfo,
    scoring: &ScoringConfig,
    locks: &LockSet,
) -> Vec<Picks> {
    let mut candidates: Vec<Vec<TeamLock>> = Vec::with_capacity(1 + 64 * 6);
    candidates.push(locks.locks.clone());
    for (team, wins) in all_moves() {
        let mut with_extra = locks.locks.clone();
        with_extra.push(TeamLock {
            team_index: team,
            wins_required: wins,
        });
        candidates.push(with_extra);
    }

    let solved: Vec<Option<Picks>> = candidates
        .par_iter()
        .map(|constraint| {
            exact::solve(tournament, scoring, constraint)
                .ok()
                .map(|s| Picks::from_winners(tournament, &s.bracket.winner_indices()))
        })
        .collect();

    // Many constraints land on the same bracket — requiring a 1-seed to reach
    // the round of 32 usually changes nothing — so dedupe on the encoding.
    let mut seen = HashSet::new();
    let mut basis = Vec::new();
    for picks in solved.into_iter().flatten() {
        if seen.insert(picks.bits()) {
            basis.push(picks);
        }
    }
    basis
}

/// Build a `k`-entry portfolio maximizing `objective`.
pub fn optimize_for(
    tournament: &TournamentInfo,
    scoring: &ScoringConfig,
    pool: &ScenarioPool,
    locks: &LockSet,
    entries: usize,
    objective: Objective<'_>,
    verbose: bool,
) -> PortfolioPlan {
    assert!(entries > 0, "a portfolio needs at least one entry");

    let basis = exact_basis(tournament, scoring, locks);
    if verbose {
        println!(
            "Basis: {} distinct conditionally optimal brackets.",
            basis.len()
        );
    }
    let prepared_basis: Vec<ScoredPicks> = basis.iter().map(|p| pool.prepare(p)).collect();

    // ---- Greedy selection -------------------------------------------------
    // Take the bracket that adds most to the portfolio's best-ball score, fold
    // it into the running baseline, repeat. The baseline is what makes this
    // affordable: each candidate costs one bracket's scoring, not the whole
    // portfolio's.
    let mut chosen: Vec<usize> = Vec::with_capacity(entries);
    let mut baseline = vec![0.0f32; pool.size()];
    let mut score = 0.0;
    let mut single_entry = 0.0;

    for slot in 0..entries.min(basis.len()) {
        let (best, best_score) =
            argmax_against(pool, &prepared_basis, &baseline, &chosen, objective);
        chosen.push(best);
        pool.absorb_into(&prepared_basis[best], &mut baseline);
        if verbose {
            println!(
                "  entry {}: {} {:.5} (+{:.5})",
                slot + 1,
                objective.name(),
                best_score,
                best_score - score
            );
        }
        score = best_score;
        if slot == 0 {
            single_entry = best_score;
        }
    }

    let mut portfolio: Vec<Picks> = chosen.iter().map(|&i| basis[i]).collect();
    // A basis smaller than the requested portfolio can only happen with locks
    // so tight that few brackets are legal; fill from what exists.
    while portfolio.len() < entries {
        portfolio.push(basis[portfolio.len() % basis.len()]);
    }
    let greedy_score = objective.value_of(pool, &pool.prepare_all(&portfolio));

    // ---- Coordinate ascent ------------------------------------------------
    let (portfolio, sweeps) =
        polish(tournament, pool, locks, portfolio, &basis, objective, verbose);

    let final_score = objective.value_of(pool, &pool.prepare_all(&portfolio));

    PortfolioPlan {
        entries: portfolio,
        best_ball: final_score,
        single_entry,
        basis_size: basis.len(),
        sweeps,
        polish_gain: final_score - greedy_score,
    }
}

/// Index of the candidate that maximizes best-ball once added to `baseline`,
/// skipping anything already chosen.
fn argmax_against(
    pool: &ScenarioPool,
    candidates: &[ScoredPicks],
    baseline: &[f32],
    exclude: &[usize],
    objective: Objective<'_>,
) -> (usize, f64) {
    let scored: Vec<(usize, f64)> = candidates
        .par_iter()
        .enumerate()
        .filter(|(i, _)| !exclude.contains(i))
        .map(|(i, c)| (i, objective.value(pool, c, baseline)))
        .collect();

    // Ties broken by index so the result does not depend on thread timing.
    scored
        .into_iter()
        .fold((0usize, f64::NEG_INFINITY), |best, next| {
            if next.1 > best.1 {
                next
            } else {
                best
            }
        })
}

/// Sweep entries one at a time, replacing each with the best alternative the
/// neighbourhood offers, until a full sweep changes nothing.
fn polish(
    tournament: &TournamentInfo,
    pool: &ScenarioPool,
    locks: &LockSet,
    mut portfolio: Vec<Picks>,
    basis: &[Picks],
    objective: Objective<'_>,
    verbose: bool,
) -> (Vec<Picks>, usize) {
    // Every accepted swap strictly raises best-ball and the objective is
    // bounded, so this terminates on its own; the cap is a safety net against a
    // floating-point cycle, not an expected exit.
    const MAX_SWEEPS: usize = 32;

    let mut current = objective.value_of(pool, &pool.prepare_all(&portfolio));
    let mut sweeps = 0;
    let mut baseline = vec![0.0f32; pool.size()];

    for sweep in 0..MAX_SWEEPS {
        let mut improved = false;

        for slot in 0..portfolio.len() {
            // Per-scenario best over every entry except this one. Freezing the
            // rest is what reduces "replace one entry" to a single bracket's
            // worth of scoring per candidate.
            let others: Vec<ScoredPicks> = portfolio
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != slot)
                .map(|(_, p)| pool.prepare(p))
                .collect();
            pool.best_ball_into(&others, &mut baseline);

            let incumbent = portfolio[slot];

            // The neighbourhood is every conditionally optimal bracket plus
            // every team-round move away from the incumbent. Both sets overlap
            // heavily — forcing a team to a round it already reaches is a
            // no-op, and different moves often land on the same bracket — so
            // dedupe on the encoding before paying for a scenario sweep.
            let mut seen: HashSet<u64> = HashSet::with_capacity(basis.len() + 64 * 6);
            seen.insert(incumbent.bits());
            let mut candidates: Vec<Picks> = Vec::with_capacity(basis.len() + 64 * 6);

            for picks in basis {
                if seen.insert(picks.bits()) {
                    candidates.push(*picks);
                }
            }
            for (team, wins) in all_moves() {
                let mut moved = incumbent.forced_to_round(tournament, team, wins);
                locks.repair_picks(&mut moved, tournament);
                if seen.insert(moved.bits()) {
                    candidates.push(moved);
                }
            }

            let best = candidates
                .par_iter()
                .map(|p| objective.value(pool, &pool.prepare(p), &baseline))
                .enumerate()
                // Ties go to the lowest index, so the answer does not depend on
                // how rayon happened to schedule the work.
                .max_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap()
                        .reverse()
                        .then(a.0.cmp(&b.0).reverse())
                });

            if let Some((index, score)) = best {
                // Accept only a real gain, so floating-point noise cannot make
                // the sweep loop forever swapping equivalent brackets.
                if score > current + 1e-12 {
                    portfolio[slot] = candidates[index];
                    current = score;
                    improved = true;
                }
            }
        }

        if !improved {
            break;
        }
        sweeps = sweep + 1;
        if verbose {
            println!("  sweep {}: {} {:.5}", sweeps, objective.name(), current);
        }
    }

    (portfolio, sweeps)
}

/// Best-ball score of a portfolio on a pool it was *not* optimized against.
///
/// Optimizing against a finite sample of tournaments will always flatter itself
/// a little — some of the gain is fitting the sample rather than the model. An
/// independently seeded pool measures how much of it is real, which is the only
/// way to tell a better portfolio from a better-overfitted one.
pub fn holdout_score(
    tournament: &TournamentInfo,
    scoring: &ScoringConfig,
    entries: &[Picks],
    size: usize,
    seed: u64,
) -> f64 {
    let holdout = ScenarioPool::new(tournament, size, scoring, seed);
    holdout.par_best_ball_mean(&holdout.prepare_all(entries))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::tests::tournament;

    fn pool(t: &TournamentInfo, scoring: &ScoringConfig) -> ScenarioPool {
        ScenarioPool::new(t, 20_000, scoring, 0xD1CE)
    }

    #[test]
    fn every_basis_bracket_is_legal_and_distinct() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let basis = exact_basis(&t, &scoring, &LockSet::default());

        assert!(basis.len() > 20, "basis was only {} brackets", basis.len());
        let mut seen = HashSet::new();
        for picks in &basis {
            assert!(seen.insert(picks.bits()), "duplicate basis bracket");
            crate::bracket::tests::assert_legal(&Bracket::from_picks(&t, picks, Some(&scoring)));
        }
    }

    #[test]
    fn the_basis_contains_the_unconstrained_optimum() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let best = exact::solve(&t, &scoring, &[]).unwrap();
        let optimum = Picks::from_winners(&t, &best.bracket.winner_indices());
        let basis = exact_basis(&t, &scoring, &LockSet::default());
        assert!(basis.iter().any(|p| p.bits() == optimum.bits()));
    }

    #[test]
    fn a_bigger_portfolio_is_never_worse() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);
        let locks = LockSet::default();

        let mut previous = 0.0;
        for k in 1..=4 {
            let plan = optimize_for(&t, &scoring, &p, &locks, k, Objective::BestBall, false);
            assert_eq!(plan.entries.len(), k);
            assert!(
                plan.best_ball >= previous - 1e-9,
                "{} entries scored {:.4}, {} entries scored {:.4}",
                k,
                plan.best_ball,
                k - 1,
                previous
            );
            previous = plan.best_ball;
        }
    }

    #[test]
    fn a_single_entry_portfolio_is_the_exact_optimum() {
        // With one entry, best-ball is just expected score, whose maximizer the
        // DP already knows. The search must not do worse than that.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);

        let plan = optimize_for(&t, &scoring, &p, &LockSet::default(), 1, Objective::BestBall, false);
        let exact_best = exact::solve(&t, &scoring, &[]).unwrap();
        let sampled_optimum = p.par_mean_score(
            &p.prepare(&Picks::from_winners(&t, &exact_best.bracket.winner_indices())),
        );
        assert!(
            plan.best_ball >= sampled_optimum - 1e-9,
            "search {:.4} vs exact {:.4}",
            plan.best_ball,
            sampled_optimum
        );
    }

    #[test]
    fn polishing_never_lowers_the_score() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);
        let plan = optimize_for(&t, &scoring, &p, &LockSet::default(), 5, Objective::BestBall, false);
        assert!(
            plan.polish_gain >= -1e-9,
            "local search lost {:.6}",
            plan.polish_gain
        );
    }

    #[test]
    fn the_result_is_deterministic() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);
        let first = optimize_for(&t, &scoring, &p, &LockSet::default(), 4, Objective::BestBall, false);
        for _ in 0..3 {
            let again = optimize_for(&t, &scoring, &p, &LockSet::default(), 4, Objective::BestBall, false);
            let a: Vec<u64> = first.entries.iter().map(|e| e.bits()).collect();
            let b: Vec<u64> = again.entries.iter().map(|e| e.bits()).collect();
            assert_eq!(a, b);
            assert_eq!(first.best_ball, again.best_ball);
        }
    }

    fn competition(
        t: &TournamentInfo,
        pool: &ScenarioPool,
        opponents: usize,
    ) -> crate::score::Competition {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let public = crate::field::PickPopularity::chalk(t, 1.6);
        pool.competition(4, opponents, |replicate, i| {
            let mut rng = SmallRng::seed_from_u64((replicate as u64) << 32 | i as u64);
            public.sample_entry(t, &mut rng)
        })
    }

    #[test]
    fn win_probability_is_a_probability_and_grows_with_entries() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);
        let field = competition(&t, &p, 50);
        let locks = LockSet::default();

        let mut previous = 0.0;
        for k in 1..=4 {
            let plan = optimize_for(
                &t,
                &scoring,
                &p,
                &locks,
                k,
                Objective::FirstPlace(&field),
                false,
            );
            assert!(
                plan.best_ball > 0.0 && plan.best_ball <= 1.0,
                "{} entries gave P(win) = {}",
                k,
                plan.best_ball
            );
            assert!(
                plan.best_ball >= previous - 1e-12,
                "{} entries won less often ({:.5}) than {} ({:.5})",
                k,
                plan.best_ball,
                k - 1,
                previous
            );
            previous = plan.best_ball;
        }
    }

    #[test]
    fn one_entry_against_one_opponent_is_near_a_coin_flip() {
        // Sanity anchor on the units: a single entry facing a single opponent
        // drawn from a field close to the rating model wins about half the
        // time, and cannot win more than all of it.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);
        let field = competition(&t, &p, 1);
        let plan = optimize_for(
            &t,
            &scoring,
            &p,
            &LockSet::default(),
            1,
            Objective::FirstPlace(&field),
            false,
        );
        assert!(
            plan.best_ball > 0.4 && plan.best_ball < 0.95,
            "P(beat one opponent) = {:.4}",
            plan.best_ball
        );
    }

    #[test]
    fn optimizing_for_first_place_beats_the_ev_bracket_at_winning() {
        // The whole point of the objective: against a public field, the bracket
        // that maximizes expected score is not the bracket that most often
        // finishes first, because it is the one everyone else also submitted.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);
        let field = competition(&t, &p, 200);

        let contrarian = optimize_for(
            &t,
            &scoring,
            &p,
            &LockSet::default(),
            1,
            Objective::FirstPlace(&field),
            false,
        );

        let ev_best = exact::solve(&t, &scoring, &[]).unwrap();
        let ev_picks = Picks::from_winners(&t, &ev_best.bracket.winner_indices());
        let mut profile = vec![0.0f32; p.size()];
        p.best_ball_into(&[p.prepare(&ev_picks)], &mut profile);
        let ev_win_rate = p.mean_win_share(&profile, &field);

        assert!(
            contrarian.best_ball >= ev_win_rate,
            "optimizing P(win) gave {:.5}, the EV bracket gets {:.5}",
            contrarian.best_ball,
            ev_win_rate
        );
    }

    #[test]
    fn locks_are_honoured_by_every_entry() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);

        let champion = t.teams.iter().find(|x| x.seed == 11).unwrap().team_index;
        let locks = LockSet::new(vec![TeamLock {
            team_index: champion,
            wins_required: 6,
        }]);

        let plan = optimize_for(&t, &scoring, &p, &locks, 4, Objective::BestBall, false);
        for entry in &plan.entries {
            assert_eq!(
                entry.champion(),
                champion,
                "an entry ignored the champion lock"
            );
        }
    }

    #[test]
    fn portfolio_entries_actually_differ() {
        // A portfolio of identical brackets scores exactly what one of them
        // does; the whole point is that they cover different outcomes.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = pool(&t, &scoring);
        let plan = optimize_for(&t, &scoring, &p, &LockSet::default(), 5, Objective::BestBall, false);

        let distinct: HashSet<u64> = plan.entries.iter().map(|e| e.bits()).collect();
        assert_eq!(distinct.len(), 5, "portfolio contains duplicate brackets");
        assert!(
            plan.best_ball > plan.single_entry,
            "five entries scored no better than one"
        );
    }
}
