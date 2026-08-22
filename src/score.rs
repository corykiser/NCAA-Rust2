//! Scenario pool and the scoring kernels the portfolio optimizers live in.
//!
//! Best-ball scoring — the pool payout when you may enter several brackets and
//! only your best one counts — has no closed form, so it is estimated against a
//! fixed pool of sampled tournaments. That estimate is the single hottest thing
//! the program does: one portfolio evaluation is `entries x scenarios x 63`
//! comparisons, and a genetic algorithm asks for tens of thousands of them.
//!
//! Three things make it fast here.
//!
//! **The pool is bytes, not brackets.** A scenario is 64 bytes of winner
//! indices in one contiguous allocation, so 100,000 of them is 6.4 MB that
//! streams through L2. Storing them as `Bracket`s instead cost ~120 bytes and
//! three atomic refcount bumps *per game*, which is 750 MB and 19 million
//! atomics for the same pool.
//!
//! **Points are resolved before the loop, not inside it.** Scoring a pick is
//! `points(round, seed(winner))`, a dependent double lookup. Since a
//! candidate's winners are fixed across the whole pool, its 63 point values are
//! resolved once into a flat `f32` array; the inner loop is then a compare and
//! a masked add with nothing to chase.
//!
//! **The accumulator is split into lanes.** `f32` addition is not associative,
//! so a single running total forces the compiler to keep the loop scalar and
//! serially dependent. Eight independent lanes summed at the end let it emit
//! one vector op per group of eight games — and fix the summation order, so
//! results stay bit-for-bit reproducible.
//!
//! On top of that the incremental form matters more than any of it. When a
//! portfolio is fixed except for one entry, its per-scenario best is a constant
//! vector; [`ScenarioPool::mean_max_with`] scores the one candidate against
//! that baseline instead of rescoring the whole portfolio, which turns an
//! `O(entries)` evaluation into an `O(1)` one.

use crate::bracket::ScoringConfig;
use crate::ingest::TournamentInfo;
use crate::picks::{Picks, PADDED_GAMES};
use crate::tree::{NUM_GAMES, NUM_ROUNDS, NUM_TEAMS, ROUND_OF};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Independent accumulator lanes in the portable kernel. Four `f32`s is one
/// SSE register; the loop is correct, and identically ordered, at any width.
const LANES: usize = 4;

/// Points a correct pick earns, resolved down to (round, team).
///
/// The scoring rules are expressed per round and per *seed*, but a team's seed
/// never changes, so collapsing the two lookups into one removes a dependent
/// load from the inner loop.
#[derive(Debug, Clone)]
pub struct PointTable {
    by_round_team: [[f32; NUM_TEAMS]; NUM_ROUNDS],
}

impl PointTable {
    pub fn new(tournament: &TournamentInfo, scoring: &ScoringConfig) -> PointTable {
        let table = crate::bracket::ScoreTable::new(scoring);
        let mut by_round_team = [[0.0f32; NUM_TEAMS]; NUM_ROUNDS];
        for round in 0..NUM_ROUNDS {
            for team in 0..NUM_TEAMS {
                by_round_team[round][team] =
                    table.get(round, tournament.seed_of[team]) as f32;
            }
        }
        PointTable { by_round_team }
    }

    /// Points for correctly picking `team` to win a game in `round`.
    #[inline(always)]
    pub fn get(&self, round: usize, team: u8) -> f32 {
        self.by_round_team[round][team as usize]
    }
}

/// A candidate bracket with its per-game payouts already resolved.
///
/// Built once per candidate and reused across every scenario in the pool.
#[derive(Debug, Clone, Copy)]
pub struct ScoredPicks {
    winners: [u8; PADDED_GAMES],
    /// Points earned for each game if the pick is right. The padding lane is
    /// zero, so it can never contribute however the lanes line up.
    points: [f32; PADDED_GAMES],
}

impl ScoredPicks {
    pub fn new(picks: &Picks, points: &PointTable) -> ScoredPicks {
        ScoredPicks::from_winners(picks.winners(), points)
    }

    /// As `new`, from bare winner indices — for callers holding a full
    /// [`Bracket`](crate::bracket::Bracket) rather than a `Picks`.
    pub fn from_winners(winners: &[u8; NUM_GAMES], points: &PointTable) -> ScoredPicks {
        let mut lanes = [u8::MAX; PADDED_GAMES];
        lanes[..NUM_GAMES].copy_from_slice(winners);
        let mut p = [0.0f32; PADDED_GAMES];
        for game in 0..NUM_GAMES {
            p[game] = points.get(ROUND_OF[game], winners[game]);
        }
        ScoredPicks {
            winners: lanes,
            points: p,
        }
    }

    /// The best this bracket could possibly score: every pick correct.
    ///
    /// Only the kernel's self-scoring test needs this, but it is the natural
    /// upper bound on `row_score` and belongs next to it.
    #[cfg(test)]
    pub fn perfect_score(&self) -> f64 {
        self.points.iter().map(|&p| p as f64).sum()
    }
}

/// Score one candidate against one scenario.
///
/// The obvious spelling of this — `if the winners match { acc += points }` — is
/// a trap. LLVM compiles it to a *branch* per game, and whether a pick matches a
/// sampled tournament is close to a coin flip, so one scenario costs sixty
/// branch mispredictions. That single detail was most of the scoring cost; the
/// arithmetic is trivial, the branch predictor is not.
///
/// Both implementations below are branchless. The SSE2 one compares sixteen
/// games at a time, sign-extends the byte mask up to `f32` lanes with three
/// unpack levels, and masks the payouts before adding — no data-dependent
/// control flow at all. SSE2 is part of the x86-64 baseline, so this needs no
/// runtime feature detection.
#[inline(always)]
fn row_score(candidate: &ScoredPicks, row: &[u8; PADDED_GAMES]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is guaranteed on every x86-64 target, and both operands
        // are fixed 64-element arrays read through unaligned loads.
        unsafe { row_score_sse2(candidate, row) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        row_score_portable(candidate, row)
    }
}

/// Branchless scalar fallback: the comparison becomes an all-ones or all-zeros
/// integer mask applied to the payout's bit pattern, which no compiler can turn
/// back into a branch.
#[inline(always)]
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
fn row_score_portable(candidate: &ScoredPicks, row: &[u8; PADDED_GAMES]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    for group in 0..PADDED_GAMES / LANES {
        let base = group * LANES;
        for (lane, acc) in lanes.iter_mut().enumerate() {
            let g = base + lane;
            let hit = ((candidate.winners[g] == row[g]) as u32).wrapping_neg();
            *acc += f32::from_bits(candidate.points[g].to_bits() & hit);
        }
    }
    let mut total = 0.0f32;
    for acc in lanes {
        total += acc;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn row_score_sse2(candidate: &ScoredPicks, row: &[u8; PADDED_GAMES]) -> f32 {
    use std::arch::x86_64::*;

    let winners = candidate.winners.as_ptr();
    let points = candidate.points.as_ptr();
    let scenario = row.as_ptr();

    let mut acc = [_mm_setzero_ps(); 4];

    for block in 0..4 {
        let offset = block * 16;
        let mine = _mm_loadu_si128(winners.add(offset) as *const __m128i);
        let theirs = _mm_loadu_si128(scenario.add(offset) as *const __m128i);
        // 0xFF in every byte whose pick was right.
        let hit = _mm_cmpeq_epi8(mine, theirs);

        // Widen the byte mask to four dword masks. Unpacking the mask with
        // itself doubles each byte, which is exactly a sign extension when
        // every byte is already 0x00 or 0xFF.
        let words_lo = _mm_unpacklo_epi8(hit, hit);
        let words_hi = _mm_unpackhi_epi8(hit, hit);
        let masks = [
            _mm_unpacklo_epi16(words_lo, words_lo),
            _mm_unpackhi_epi16(words_lo, words_lo),
            _mm_unpacklo_epi16(words_hi, words_hi),
            _mm_unpackhi_epi16(words_hi, words_hi),
        ];

        for (quad, mask) in masks.iter().enumerate() {
            let p = _mm_loadu_ps(points.add(offset + quad * 4));
            acc[quad] = _mm_add_ps(acc[quad], _mm_and_ps(p, _mm_castsi128_ps(*mask)));
        }
    }

    // Fixed reduction order, so results are reproducible run to run.
    let pairs = _mm_add_ps(_mm_add_ps(acc[0], acc[1]), _mm_add_ps(acc[2], acc[3]));
    let mut out = [0.0f32; 4];
    _mm_storeu_ps(out.as_mut_ptr(), pairs);
    out[0] + out[1] + out[2] + out[3]
}

/// A fixed sample of tournament outcomes to score candidates against.
pub struct ScenarioPool {
    /// `PADDED_GAMES` bytes of winner indices per scenario, contiguous.
    rows: Vec<u8>,
    size: usize,
    points: PointTable,
}

impl ScenarioPool {
    /// Sample `size` tournaments from the rating model.
    ///
    /// `seed` fixes the pool exactly, which is what lets two optimizer runs be
    /// compared without sampling noise between them.
    pub fn new(
        tournament: &TournamentInfo,
        size: usize,
        scoring: &ScoringConfig,
        seed: u64,
    ) -> ScenarioPool {
        assert!(size > 0, "a scenario pool needs at least one scenario");

        // One RNG per chunk, seeded from the pool seed and the chunk index, so
        // the pool is identical however many threads happen to build it.
        const CHUNK: usize = 1024;
        let mut rows = vec![0u8; size * PADDED_GAMES];
        rows.par_chunks_mut(CHUNK * PADDED_GAMES)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let mut rng = SmallRng::seed_from_u64(seed ^ (chunk_index as u64).wrapping_mul(
                    0x9E37_79B9_7F4A_7C15,
                ));
                for row in chunk.chunks_exact_mut(PADDED_GAMES) {
                    let picks = Picks::sample(tournament, &mut rng);
                    row.copy_from_slice(picks.winner_lanes());
                }
            });

        ScenarioPool {
            rows,
            size,
            points: PointTable::new(tournament, scoring),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn points(&self) -> &PointTable {
        &self.points
    }

    /// Resolve a bracket's payouts against this pool's scoring rules.
    #[inline]
    pub fn prepare(&self, picks: &Picks) -> ScoredPicks {
        ScoredPicks::new(picks, &self.points)
    }

    pub fn prepare_all(&self, picks: &[Picks]) -> Vec<ScoredPicks> {
        picks.iter().map(|p| self.prepare(p)).collect()
    }

    #[inline(always)]
    fn row(&self, index: usize) -> &[u8; PADDED_GAMES] {
        as_row(&self.rows[index * PADDED_GAMES..(index + 1) * PADDED_GAMES])
    }

    /// Mean score of one bracket over the whole pool.
    ///
    /// Serial: call this from inside a parallel loop over candidates. Nesting
    /// rayon inside rayon here costs more in scheduling than the 63-byte inner
    /// loop can ever win back.
    pub fn mean_score(&self, candidate: &ScoredPicks) -> f64 {
        let mut total = 0.0f64;
        for row in self.rows.chunks_exact(PADDED_GAMES).map(as_row) {
            total += row_score(candidate, row) as f64;
        }
        total / self.size as f64
    }

    /// `mean_score` for a single candidate, spread across threads.
    pub fn par_mean_score(&self, candidate: &ScoredPicks) -> f64 {
        let total: f64 = self
            .rows
            .par_chunks(chunk_rows(self.size) * PADDED_GAMES)
            .map(|block| {
                let mut sub = 0.0f64;
                for row in block.chunks_exact(PADDED_GAMES).map(as_row) {
                    sub += row_score(candidate, row) as f64;
                }
                sub
            })
            .sum();
        total / self.size as f64
    }

    /// Per-scenario best score across a portfolio — the best-ball profile.
    ///
    /// Every incremental evaluation is measured against one of these, so it is
    /// computed once per frozen portfolio rather than once per candidate.
    pub fn best_ball_into(&self, portfolio: &[ScoredPicks], out: &mut [f32]) {
        assert_eq!(out.len(), self.size);
        out.fill(f32::NEG_INFINITY);
        if portfolio.is_empty() {
            out.fill(0.0);
            return;
        }
        out.par_chunks_mut(chunk_rows(self.size))
            .enumerate()
            .for_each(|(block, slots)| {
                let first = block * chunk_rows(self.size);
                for (i, slot) in slots.iter_mut().enumerate() {
                    let row = self.row(first + i);
                    let mut best = f32::NEG_INFINITY;
                    for candidate in portfolio {
                        let s = row_score(candidate, row);
                        if s > best {
                            best = s;
                        }
                    }
                    *slot = best;
                }
            });
    }

    /// Expected best-ball payout of a whole portfolio.
    pub fn best_ball_mean(&self, portfolio: &[ScoredPicks]) -> f64 {
        if portfolio.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f64;
        for row in self.rows.chunks_exact(PADDED_GAMES).map(as_row) {
            let mut best = f32::NEG_INFINITY;
            for candidate in portfolio {
                let s = row_score(candidate, row);
                if s > best {
                    best = s;
                }
            }
            total += best as f64;
        }
        total / self.size as f64
    }

    /// `best_ball_mean` spread across threads.
    pub fn par_best_ball_mean(&self, portfolio: &[ScoredPicks]) -> f64 {
        if portfolio.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .rows
            .par_chunks(chunk_rows(self.size) * PADDED_GAMES)
            .map(|block| {
                let mut sub = 0.0f64;
                for row in block.chunks_exact(PADDED_GAMES).map(as_row) {
                    let mut best = f32::NEG_INFINITY;
                    for candidate in portfolio {
                        let s = row_score(candidate, row);
                        if s > best {
                            best = s;
                        }
                    }
                    sub += best as f64;
                }
                sub
            })
            .sum();
        total / self.size as f64
    }

    /// Expected best-ball payout of `baseline` once `candidate` is added to it.
    ///
    /// This is the whole point of keeping a baseline around: swapping one entry
    /// of a `k`-bracket portfolio costs one bracket's worth of scoring instead
    /// of `k`, which is what makes an exhaustive coordinate sweep affordable.
    pub fn mean_max_with(&self, candidate: &ScoredPicks, baseline: &[f32]) -> f64 {
        debug_assert_eq!(baseline.len(), self.size);
        let mut total = 0.0f64;
        for (row, &base) in self.rows.chunks_exact(PADDED_GAMES).map(as_row).zip(baseline.iter()) {
            let s = row_score(candidate, row);
            total += if s > base { s as f64 } else { base as f64 };
        }
        total / self.size as f64
    }

    /// Fold `candidate`'s scores into an existing best-ball profile.
    pub fn absorb_into(&self, candidate: &ScoredPicks, baseline: &mut [f32]) {
        debug_assert_eq!(baseline.len(), self.size);
        for (row, base) in self.rows.chunks_exact(PADDED_GAMES).map(as_row).zip(baseline.iter_mut()) {
            let s = row_score(candidate, row);
            if s > *base {
                *base = s;
            }
        }
    }
}

/// What the rest of the pool scores, scenario by scenario.
///
/// For a pool that pays the top finisher, a portfolio's value is
/// `P(one of my entries finishes first)`, which needs the competition's scores
/// as well as your own. Reduced to what the objective actually reads, that is
/// two numbers per scenario: the best score anyone else posted, and how many of
/// them posted it — the second because finishing level with `n` other entries
/// is worth `1/(n+1)` of first place, and that tie penalty is the entire reason
/// picking the same champion as a fifth of the field is expensive.
///
/// The field is drawn several times over. Your real pool is one fixed set of
/// entries that you cannot see, so a single draw would optimize against one
/// arbitrary guess at it; averaging over replicates prices in the fact that you
/// do not know who you are playing.
pub struct Competition {
    /// `best[replicate * size + scenario]`.
    best: Vec<f32>,
    /// Opponents tied at `best`, same indexing.
    ties: Vec<u32>,
    size: usize,
    replicates: usize,
    /// Opposing entries per replicate. Reported, not read by the kernel.
    #[allow(dead_code)]
    pub opponents: usize,
}

impl Competition {
    /// Share of first place a score of `mine` takes in one scenario.
    #[inline(always)]
    fn share(&self, scenario: usize, mine: f32) -> f64 {
        let mut total = 0.0f64;
        for replicate in 0..self.replicates {
            let i = replicate * self.size + scenario;
            let best = self.best[i];
            total += if mine > best {
                1.0
            } else if mine == best {
                1.0 / (1.0 + self.ties[i] as f64)
            } else {
                0.0
            };
        }
        total / self.replicates as f64
    }

    /// Mean top opponent score, for reporting.
    pub fn mean_top_score(&self) -> f64 {
        self.best.iter().map(|&v| v as f64).sum::<f64>() / self.best.len() as f64
    }
}

impl ScenarioPool {
    /// Score a sampled public field down to per-scenario order statistics.
    ///
    /// `draw(replicate, index)` supplies one opposing entry. Cost is
    /// `replicates * opponents * scenarios` row scores, paid once — after which
    /// every candidate evaluation is `O(replicates)` per scenario, the same
    /// shape as the best-ball baseline.
    pub fn competition(
        &self,
        replicates: usize,
        opponents: usize,
        mut draw: impl FnMut(usize, usize) -> Picks,
    ) -> Competition {
        assert!(replicates > 0 && opponents > 0);

        let mut best = vec![f32::NEG_INFINITY; replicates * self.size];
        let mut ties = vec![0u32; replicates * self.size];

        for replicate in 0..replicates {
            let field: Vec<ScoredPicks> = (0..opponents)
                .map(|i| self.prepare(&draw(replicate, i)))
                .collect();

            let offset = replicate * self.size;
            let rows = chunk_rows(self.size);
            best[offset..offset + self.size]
                .par_chunks_mut(rows)
                .zip(ties[offset..offset + self.size].par_chunks_mut(rows))
                .enumerate()
                .for_each(|(block, (best_block, ties_block))| {
                    let first = block * rows;
                    for (i, (top, count)) in
                        best_block.iter_mut().zip(ties_block.iter_mut()).enumerate()
                    {
                        let row = self.row(first + i);
                        let mut high = f32::NEG_INFINITY;
                        let mut seen = 0u32;
                        for entry in &field {
                            let s = row_score(entry, row);
                            if s > high {
                                high = s;
                                seen = 1;
                            } else if s == high {
                                seen += 1;
                            }
                        }
                        *top = high;
                        *count = seen;
                    }
                });
        }

        Competition {
            best,
            ties,
            size: self.size,
            replicates,
            opponents,
        }
    }

    /// Expected share of first place for a portfolio already reduced to its
    /// per-scenario best.
    pub fn mean_win_share(&self, mine_best: &[f32], competition: &Competition) -> f64 {
        debug_assert_eq!(mine_best.len(), self.size);
        let total: f64 = mine_best
            .iter()
            .enumerate()
            .map(|(s, &mine)| competition.share(s, mine))
            .sum();
        total / self.size as f64
    }

    /// Expected share of first place once `candidate` joins a portfolio whose
    /// per-scenario best is `baseline`.
    ///
    /// Same incremental trick as `mean_max_with`: the frozen entries collapse
    /// to one number per scenario, so trying a replacement costs one bracket's
    /// scoring rather than the portfolio's.
    pub fn mean_win_share_with(
        &self,
        candidate: &ScoredPicks,
        baseline: &[f32],
        competition: &Competition,
    ) -> f64 {
        debug_assert_eq!(baseline.len(), self.size);
        let mut total = 0.0f64;
        for (scenario, (row, &base)) in self
            .rows
            .chunks_exact(PADDED_GAMES)
            .map(as_row)
            .zip(baseline.iter())
            .enumerate()
        {
            let mine = row_score(candidate, row).max(base);
            total += competition.share(scenario, mine);
        }
        total / self.size as f64
    }
}

/// View one stride of the pool as a fixed-size scenario row.
#[inline(always)]
fn as_row(chunk: &[u8]) -> &[u8; PADDED_GAMES] {
    chunk
        .try_into()
        .expect("scenarios are stored in fixed-size strides")
}

/// Rows per parallel block: big enough that rayon's per-task overhead is noise
/// next to the work, small enough to keep every core busy.
fn chunk_rows(size: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (size / (threads * 4)).max(256).min(size.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bracket::{Bracket, FastBracket, ScoreTable};
    use crate::ingest::tests::tournament;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn pool(size: usize) -> (TournamentInfo, ScoringConfig, ScenarioPool) {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = ScenarioPool::new(&t, size, &scoring, 0xA11CE);
        (t, scoring, p)
    }

    #[test]
    fn the_kernel_agrees_with_the_reference_scorer() {
        // The lane-split f32 kernel must match the straightforward f64 scorer
        // it replaced, for every pair in a decent sample.
        let (t, scoring, p) = pool(64);
        let table = ScoreTable::new(&scoring);
        let mut rng = SmallRng::seed_from_u64(99);

        for _ in 0..200 {
            let picks = Picks::sample(&t, &mut rng);
            let bracket = Bracket::new_from_binary_bits(&t, picks.bits(), Some(&scoring));
            let fast = FastBracket::from_bracket(&bracket);
            let prepared = p.prepare(&picks);

            for i in 0..p.size() {
                let scenario_picks = Picks::from_winners(
                    &t,
                    p.row(i)[..NUM_GAMES].try_into().unwrap(),
                );
                let scenario = FastBracket::from_bracket(&Bracket::new_from_binary_bits(
                    &t,
                    scenario_picks.bits(),
                    Some(&scoring),
                ));
                let reference = fast.score_against(&scenario, &table);
                let actual = row_score(&prepared, p.row(i)) as f64;
                assert!(
                    (reference - actual).abs() < 1e-6,
                    "scenario {}: kernel {} vs reference {}",
                    i,
                    actual,
                    reference
                );
            }
        }
    }

    #[test]
    fn a_bracket_scored_against_itself_earns_its_maximum() {
        let (t, _, p) = pool(4);
        let mut rng = SmallRng::seed_from_u64(3);
        let picks = Picks::sample(&t, &mut rng);
        let prepared = p.prepare(&picks);
        let mut row = [0u8; PADDED_GAMES];
        row.copy_from_slice(picks.winner_lanes());
        assert!((row_score(&prepared, &row) as f64 - prepared.perfect_score()).abs() < 1e-6);
    }

    #[test]
    fn best_ball_dominates_every_entry_and_the_baseline_form_agrees() {
        let (t, _, p) = pool(2_000);
        let mut rng = SmallRng::seed_from_u64(5);
        let entries: Vec<Picks> = (0..4).map(|_| Picks::sample(&t, &mut rng)).collect();
        let prepared = p.prepare_all(&entries);

        let best_ball = p.best_ball_mean(&prepared);
        for one in &prepared {
            assert!(
                best_ball >= p.mean_score(one) - 1e-9,
                "best ball {} is below a single entry {}",
                best_ball,
                p.mean_score(one)
            );
        }

        // Incremental form: baseline over the first three, plus the fourth.
        let mut baseline = vec![0.0f32; p.size()];
        p.best_ball_into(&prepared[..3], &mut baseline);
        let incremental = p.mean_max_with(&prepared[3], &baseline);
        assert!(
            (incremental - best_ball).abs() < 1e-9,
            "incremental {} vs full {}",
            incremental,
            best_ball
        );

        // And absorbing the fourth must reproduce the full profile.
        let mut absorbed = baseline.clone();
        p.absorb_into(&prepared[3], &mut absorbed);
        let mut full = vec![0.0f32; p.size()];
        p.best_ball_into(&prepared, &mut full);
        assert_eq!(absorbed, full);
    }

    #[test]
    fn parallel_and_serial_kernels_agree_exactly() {
        let (t, _, p) = pool(5_000);
        let mut rng = SmallRng::seed_from_u64(23);
        let entries: Vec<Picks> = (0..3).map(|_| Picks::sample(&t, &mut rng)).collect();
        let prepared = p.prepare_all(&entries);
        assert!((p.mean_score(&prepared[0]) - p.par_mean_score(&prepared[0])).abs() < 1e-9);
        assert!((p.best_ball_mean(&prepared) - p.par_best_ball_mean(&prepared)).abs() < 1e-9);
    }

    #[test]
    fn a_pool_is_reproducible_from_its_seed() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let a = ScenarioPool::new(&t, 3_000, &scoring, 42);
        let b = ScenarioPool::new(&t, 3_000, &scoring, 42);
        let c = ScenarioPool::new(&t, 3_000, &scoring, 43);
        assert_eq!(a.rows, b.rows);
        assert_ne!(a.rows, c.rows);
    }

    #[test]
    fn sampled_scenarios_are_legal_brackets() {
        let (t, _, p) = pool(500);
        for i in 0..p.size() {
            let winners: &[u8; NUM_GAMES] = p.row(i)[..NUM_GAMES].try_into().unwrap();
            for game in 32..NUM_GAMES {
                let [c0, c1] = crate::tree::CHILDREN[game];
                assert!(winners[game] == winners[c0] || winners[game] == winners[c1]);
            }
            let _ = Picks::from_winners(&t, winners);
        }
    }

    #[test]
    fn the_pool_mean_tracks_the_exact_expected_value() {
        // A bracket's mean score against sampled tournaments must converge to
        // the exact expected value the DP optimizes, or the two halves of the
        // program are optimizing different things.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let p = ScenarioPool::new(&t, 200_000, &scoring, 2024);
        let solution = crate::exact::solve(&t, &scoring, &[]).unwrap();
        let picks = Picks::from_winners(&t, &solution.bracket.winner_indices());
        let sampled = p.par_mean_score(&p.prepare(&picks));
        assert!(
            (sampled - solution.expected_value).abs() < 1.5,
            "sampled {:.3} vs exact {:.3}",
            sampled,
            solution.expected_value
        );
    }
}
