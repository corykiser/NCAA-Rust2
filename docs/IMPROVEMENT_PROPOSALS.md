# NCAA-Rust2 — Improvement Proposals

Review of the codebase at `e48c0a0`, with behavior verified by running the release
binary against `--source csv`. Ordered by impact.

> **Status.** Tier 0 (§0.1-0.5) and §1.1, plus §2.5 and part of §2.1, are
> **implemented**. Each is marked below. Everything else stands as proposed.

---

## Tier 0 — Bugs that silently produce wrong output

These are the priority: each one produces confident, plausible-looking output that
is wrong. No error, no warning.

### 0.1 `--lock-team` produces impossible brackets and the lock is not honored — **FIXED**

`ConstrainedBracketBuilder::apply_constraint` (`src/portfolio.rs:180`) overwrites
`bracket.games[idx].winner` in place. It does **not** propagate the new winner into
the downstream games' `team1`/`team2`, does **not** update `bracket.binary`, and does
**not** recompute `winnerprob` for the overwritten game.

Reproduction:

```
$ ./target/release/ncaa --source csv --lock-team "Alabama:FinalFour" --generations 5
Elite 8
1 Purdue
1 Kansas
1 Alabama     <- lock applied here
2 Texas

Final Four
9 West Virginia   <- did not win any Elite 8 game
1 Purdue
```

West Virginia advances to the Final Four without appearing in the Elite 8. The Final
Four game still holds the pre-override winner. The bracket is not a legal bracket.

Three separate consequences:
- `binary` and `games` now disagree. Every later `mutate()` reads `binary`, so the
  lock is discarded on the first mutation regardless.
- `winnerprob` on the overwritten game belongs to the *previous* winner, so
  `recalculate_bracket_stats` reports an EV for a bracket that doesn't exist.
- `AdvancementRound::wins_required()` semantics ("wins needed to reach round R") are
  right, but nothing downstream of the write is.

**Fix:** make `binary` the single source of truth. Apply constraints by editing the
63-bit vector, then call `Bracket::new_from_binary` to rebuild. This makes
propagation automatic and the result legal by construction. `apply_constraint_to_binary`
(`src/portfolio.rs:236`) is the start of this and is currently a stub that handles
round 1 only; its own comment admits the limitation. Finish that function and delete
`apply_constraint`.

*Implemented. `ConstrainedBracketBuilder` now edits the bit vector via `LockSet`
and rebuilds, and verifies the constraints before returning. `LockSet::repair` is
applied to every candidate the optimizers produce, so locks also survive mutation
and crossover — they previously did not.*

### 0.2 The constrained optimizer's accept test can never fire — **FIXED**

`src/main.rs:683`:

```rust
if batch.batch_score > bracket.score {
```

`batch_score` is the mean score against random simulations (~120–230 with default
scoring). `bracket.score` is the bracket's *perfect* score — the sum of all 63 win
values (~500). The comparison is between two different quantities and the right-hand
side is a constant that is always larger.

Verified: EV is byte-identical at generation 0 and after all generations.

```
$ ./target/release/ncaa --source csv --lock-team "Alabama:FinalFour" --generations 10
Expected Value: 276.11        <- initial
Generation 0: EV = 276.11
Expected Value: 276.1133386173198   <- final, unchanged
```

`--generations` in constrained mode burns CPU and does nothing. It should compare
against the incumbent's own batch score.

*Implemented. The separate constrained path is gone; locks now flow into whichever
optimizer runs, and the hill climber compares against the incumbent's own batch
score. It now improves 217 -> 235 over 100 generations where it previously moved
not at all.*

### 0.3 Smart mutation does not force the team to advance — **FIXED**

`TeamRoundMutator::force_team_to_round` is the core operator — it is used by every
mutation and every crossover in `GeneticAlgorithm`, `WholePortfolioGA`, and
`HybridOptimizer`. For rounds 2–4 it delegates to
`should_be_lower_seed_won_winner` (`src/ga.rs:292`), whose entire body is:

```rust
team.seed <= 8
```

This ignores the opponent. `lower_seed_won` means "the numerically lower seed of
*this matchup* advances", so the correct bit depends on who the team is actually
playing — which depends on earlier bits. A 6-seed facing a 3-seed needs `false`;
the function returns `false` correctly by luck, but a 6 facing an 11 needs `true`
and gets `false`. Roughly half the "forced" advances silently do the opposite.

Round 6 (`src/ga.rs:278`) is worse:

```rust
let should_win = region_idx == 2 || region_idx == 3;
```

The championship bit depends on the alphabetical order of the *two finalists'*
regions, not on one region index. The surrounding comment block visibly works
through the confusion and gives up mid-thought.

So the operator the GA is built around is a biased random bit-setter, not a
directed move. This is why the search is so flat (see 1.2).

**Fix:** see §2.1 — the encoding change makes this function trivially correct
instead of trying to patch it.

*Implemented as `TeamRoundMutator::force_index_to_round`, which walks the team's
actual path up the parent chain and computes each bit against the real opponent.
Verified for all 64 teams at all 6 depths.*

### 0.4 ELO team-name matching is a substring search over a `HashMap` — **FIXED**

`EloSystem::find_team_by_name` (`src/elo.rs:243`):

```rust
self.ratings.values().find(|r| r.team_name.to_lowercase().contains(&name_lower))
```

`HashMap::values()` has no defined order. `"Texas"` matches `"Texas A&M"`,
`"Texas Tech"`, and `"Texas"` — whichever the iterator reaches first, which differs
between runs. Every tournament team's rating is resolved through this function
(`src/ingest.rs`, `from_elo_ratings`), so a mis-resolved name silently assigns a
different school's rating.

`ConstrainedBracketBuilder::find_team` (`src/portfolio.rs:92`) has the same
substring fallback.

The shipped sample bracket makes this concrete: `sample_bracket_teams()` lists
`"Texas"` as a 7-seed in **both** East and Midwest. Since `Team::eq` compares by
name only, and `team_lookup` is keyed by `(region, seed)`, duplicate names break
`weighted_distance`, `champion_match`, and portfolio distance reporting.

**Fix:** exact match on a normalized key, backed by an explicit alias table
(`aliases.toml`) for the ESPN/NCAA/538 naming differences. On ambiguity or a miss,
fail loudly with the candidate list instead of guessing. Assert 64 distinct
`(region, seed)` pairs and 64 distinct names at ingest.

*Implemented as `src/names.rs`: normalization (including `St` -> Saint at the front
but State at the end), a built-in alias table, and three match tiers where the
first tier to produce any match decides — one hit resolves, more than one is an
error naming the candidates. Field validation now rejects duplicate slots, duplicate
names, and out-of-range seeds. The sample field's duplicated "Texas" is corrected.
With four Texas schools in the 2023 field, `Texas` and `Texas A&M` each resolve to
exactly the right one.*

### 0.5 A total data-fetch failure degrades into a uniform-random optimizer — **FIXED**

`src/main.rs`, ESPN/NCAA path: if `fetch_season` errors, it prints a warning and
substitutes `Vec::new()`. Zero games means every team keeps `DEFAULT_RATING = 1500.0`,
so every matchup is 50/50, and the optimizer proceeds to run its full pipeline on
pure noise — printing brackets, EVs, and best-ball scores that look identical in
form to a real run.

Related: `fetch_games_for_range` warns on a per-day fetch error, continues, and then
`fetch_season` **writes the partial result to the cache as if complete**. The 6-hour
staleness window then serves that truncated season for the rest of the day.

**Fix:** abort with a non-zero exit on an empty or substantially incomplete game set.
Record per-day fetch status in the cache and refuse to mark a season complete with
missing days.

*Implemented. `fetch_games_for_range` returns its failed days, `fetch_season` errors
rather than caching a partial season, and a fetch failure or an empty game set is a
hard error with a non-zero exit. `--allow-partial-data` opts out explicitly.*

---

## Tier 1 — Modeling and statistical soundness

### 1.1 The single-bracket objective is exactly solvable; the GA is unnecessary for it — **IMPLEMENTED**

This is the largest single improvement available.

The current pipeline samples 10,000 Monte Carlo tournaments and runs a GA to maximize
mean score against them. But that objective has closed form. A bracket's expected
score is

    E[score] = Σ_over_games  points(round, seed of your pick) × P(your pick actually
                             wins that game)

and `P(team t reaches round r+1)` depends only on the ELO model, not on your picks.
Those advancement probabilities are computed exactly by a DP over the bracket tree:
64 teams × 6 rounds, on the order of 64×64×6 multiply-adds — microseconds.

Given that matrix, the EV-maximizing bracket is itself a DP over the tree. Let
`f(g, t)` be the best expected points obtainable in the subtree rooted at game `g`
given you pick team `t` to win `g`:

    f(g, t) = pts(round(g), seed(t)) · P(t advances past g)
            + f(child containing t, t)
            + max over u in the other child of f(other child, u)

The answer is `max_t f(62, t)`. Total work is O(Σ_g |teams under g|) = 64·6 states.
**The provably optimal expected-value bracket, exactly, instantly, deterministically.**

What this buys:
- Replaces `--optimization-mode ga` / `legacy` / `hybrid` for the single-bracket case
  with an exact solver.
- Gives a ground-truth optimum to validate the GA against — right now there is no way
  to know whether the GA is finding anything.
- Removes all Monte Carlo noise from the single-bracket path.
- `--pool-size` becomes irrelevant for single brackets.

Monte Carlo is still needed for portfolio best-ball (max over brackets is nonlinear)
and for variance/percentile reporting. That's the right division of labor.

*Implemented as `src/exact.rs` (+ `src/advancement.rs`), and `--optimization-mode
exact` is now the default. ~3 ms including data loading, deterministic, and locks
drop into the same recurrence. The heuristic modes now print their gap to the
optimum. Verified against exhaustive enumeration: with the last two rounds scored
at zero the regions decouple, and the DP's answer matches brute force over all
2^15 outcomes of each region to 1e-9.*

Two things fell out of this. The old `expected_value` field multiplied each pick's
points by its probability of winning that game *given the matchup happened*, not
the unconditional probability — it reported ~300 for brackets whose true expected
score was ~228. It is now the real expected value and agrees with a 40,000-trial
Monte Carlo estimate. And the GA turns out to land within 0.2-0.3% of the optimum,
so the search was working; it just had no way to say so, and cost 11 s and a
different champion each run to get there.

### 1.2 The GA is currently fitting Monte Carlo noise, not signal

Three identical runs, 200 generations, 10,000-bracket pool:

| run | reported fitness | bracket EV | champion |
|-----|------------------|-----------|----------|
| 1   | 227.65           | 309.85    | 1 Houston |
| 2   | 229.04           | 297.39    | 3 Gonzaga |
| 3   | 228.49           | 328.50    | 1 Houston |

Fitness spans 0.6%. The brackets it selects span 10% in EV and disagree on the
champion. The fitness surface is flat relative to sampling noise on a 10k pool, so
the GA's selection pressure is largely acting on which brackets happened to match
the sampled scenarios. That is textbook overfitting to a fixed evaluation set — and
the pool is fixed for the entire run (`regenerate_pool` exists in `config.yaml` and
is never read anywhere).

**Fix, in order of value:**
1. Use the exact EV from §1.1 wherever the objective is linear. Noise problem gone.
2. For best-ball, hold out an independent validation pool and report the final score
   against *that*, not against the pool that was optimized. The current
   "Average score against 10000 simulations" line is an in-sample number and is
   biased upward.
3. Use common random numbers: score every candidate against the same scenarios
   (already true) *and* resample the pool on a schedule so the optimizer can't
   memorize it.
4. Report a confidence interval on the reported score. Standard error on a 10k pool
   is roughly ±0.5 points; differences smaller than that are not real.

### 1.3 No opponent model — the tool optimizes the wrong quantity for a real pool

Maximizing expected points is the correct objective only in a pool that pays out
proportional to your score. Almost no pool works that way; they pay the top finisher.
Winning requires maximizing `P(your score > every opponent's score)`, which is a
different problem: it rewards *contrarian* picks precisely because public brackets
concentrate on chalk.

There is no model of the field anywhere in the codebase. As written, the tool will
happily hand five entrants in the same office pool five near-chalk brackets with high
EV and near-zero win probability.

**Proposal:** add a public-pick distribution (ESPN "Who Picked Whom" publishes
per-team, per-round pick rates; a seed-based prior works as a fallback) and add an
objective mode:

```
--objective ev            # current behavior
--objective win-prob      # P(beat N simulated opponents drawn from public picks)
--objective ev-payout     # expected prize share, needs a payout curve
```

Under `win-prob`, generate the field once per scenario from the pick distribution and
count wins. This is the change that makes the output actually decision-useful, and it
is what the "diversity" machinery in `portfolio.rs` is groping toward without a
principled target.

### 1.4 The win-probability curve is roughly 2× too steep

`elo.rs:233`, `to_538_scale`, maps ELO to a 538-style 60–100 scale:

```rust
let normalized = ((elo - 1200.0) / 600.0).clamp(0.0, 1.0);
(60.0 + normalized * 40.0) as f32
```

Then `ProbabilityCache::new` applies `1 / (1 + 10^(-Δ · 30.464/400))` to that scaled
value. Composing the two: a 600-point ELO span is compressed into 40 scale points, so
1 scale point = 15 ELO. The effective exponent is therefore `Δ_elo · 2.03 / 400` —
**twice** the standard ELO logistic. A 100-point ELO edge, which standard ELO calls
64%, this code calls 74%.

Two independent problems:
- Over-confident probabilities compress the simulated outcome distribution, which
  makes upsets rarer than reality and biases the whole optimization toward chalk.
- The `.clamp(0.0, 1.0)` flattens every team above 1800 or below 1200 to identical
  ratings. In a strong year the top seeds become indistinguishable from each other.

**Fix:** drop the round-trip through the 538 scale entirely. Keep raw ELO on `f64`
and use the standard `Δ/400` logistic, with the slope as a single named, tunable
constant. Then calibrate it: back-test against past tournaments and check that games
predicted at 70% are won ~70% of the time (a reliability curve / Brier score). Right
now there is no calibration check of any kind.

### 1.5 ELO model gaps

Smaller, but each is a known source of rating error:
- **No preseason regression.** Everyone starts at exactly 1500 with no carry-over from
  the prior season, so November ratings are noise and early games are mispriced.
  538 regresses ~1/3 toward the mean; even a rough prior beats a flat start.
- **K-factor is averaged between the two teams** (`elo.rs`), so a veteran team's rating
  moves at a rate partly determined by its opponent's game count. Each team should use
  its own K.
- **`mov_multiplier` uses pre-game ratings without the home adjustment**, while the
  expected score does include it. Inconsistent inputs to the same update.
- **The NCAA source hard-codes `is_neutral_site: false`** (`api.rs`, `parse_ncaa_game`),
  so every neutral-court game gets a spurious 100-point home edge for one side.
- **No conference-strength prior**, so mid-majors with few cross-conference games are
  anchored near 1500 regardless of quality.

---

## Tier 2 — Representation and architecture

### 2.1 Replace the `lower_seed_won` encoding with a positional one — **PARTIALLY ADDRESSED**

The 63-bit encoding stores, per game, "did the numerically lower seed win — or for
cross-region games, the alphabetically earlier region". This is the root cause of
§0.3 and of the "bit-flip mutation is semantically broken" comments scattered through
`ga.rs`. To set one bit you must know both participants, which requires resolving all
earlier bits.

Replace it with **`left_subtree_won`**: for each game, one bit saying which of the two
feeding slots advances. Purely positional, independent of seeds, regions, and history.

Consequences:
- `force_team_to_round(team, r)` becomes: start at the team's round-1 slot, walk up
  `r` levels, set each bit to the side the team is on. Correct by construction, no
  opponent lookup, no special case for cross-region or championship games. §0.3
  disappears rather than being patched.
- Bit-flip mutation becomes meaningful: flipping one bit swaps one game's winner and
  cascades cleanly.
- Constraint application (§0.1) becomes a bit-vector write, which is what
  `apply_constraint_to_binary` was reaching for.
- Encoding and decoding stay O(63).

This is a contained change — `Game::new_from_binary*`, `Bracket::new_from_binary`, the
mutators, and `portfolio.rs`'s index helpers — and it removes more code than it adds.

*Partially addressed. The encoding itself is unchanged, but its definition now
lives in exactly one place (`TournamentInfo::bit_true_winner`), the bracket
structure lives in `src/tree.rs` instead of being re-derived per round in each
constructor, and `decode_winners` / `binary_from_winners` are the only encode and
decode paths. That was enough to make the mutator correct without changing the
representation. The positional encoding is still the better long-term shape.*

### 2.2 `Game` carries five fields nobody reads

Compiler output: *"fields `team1`, `team2`, `team1prob`, and `team2prob` are never
read."* Each `Game` holds four `Arc<Team>` and several `f64`; each `Bracket` holds 63
of them. `MonteCarloScenarios` keeps 10,000 full `Bracket`s in
`brackets: Vec<Bracket>` — also flagged never-read — purely "for debugging".

Estimated ~60–100 MB of live allocation that is never touched, plus 630k atomic
refcount operations per pool generation.

**Fix:** drop `MonteCarloScenarios::brackets` and keep only `FastBracket`. Slim `Game`
to what's actually consumed. See §3.1 for the follow-on win.

### 2.3 No `lib.rs`

Everything is `mod` declarations inside `main.rs`. Nothing is reachable from
`tests/`, from a benchmark harness, or from any other consumer. Split into
`src/lib.rs` + a thin `src/main.rs`. This is a prerequisite for §4.1 and §4.2.

### 2.4 `main.rs` is 791 lines of branching and duplicated printing

Six near-identical `println!` blocks for configuration banners, and every optimization
mode is dispatched inline. The five bracket-printing loops in `run_portfolio_mode`
are copy-pasted. Extract an `Optimizer` trait with one method, and a single reporting
function.

### 2.5 Two independent scoring configs are live at once — **FIXED**

`main.rs` builds `ScoringConfig` from the `--score-r*` / `--seed-r*` CLI flags. But
`SequentialPortfolioOptimizer::new` and `HybridOptimizer::new` call
`config.to_scoring_config()`, reading the **YAML** scoring section instead.

So `--score-r6 100` is honored under `--portfolio-strategy ga-whole` and silently
ignored under `--portfolio-strategy ga-sequential` or `--optimization-mode hybrid`.
Worse, the banner prints the CLI values in both cases.

**Fix:** one resolution point — defaults <- YAML <- CLI — producing one `ScoringConfig`
threaded everywhere. Clap's `ArgMatches::value_source` distinguishes "user passed it"
from "default", which is what makes correct layering possible.

### 2.6 Config knobs that are printed but do nothing

- `ga.smart_mutation` / `--smart-mutation` — mutation is now unconditionally
  `TeamRoundMutator`; the flag is read into config and printed in four separate
  banners, and never consulted.
- `ga.smart_mutation_rate` — never read.
- `simulation.regenerate_pool` — never read.
- `portfolio.best_ball`, `portfolio.num_brackets` — never read.
- `--diversity-weight` — only affects the two legacy strategies, which are no longer
  reachable from the CLI enum (`champion` / `diverse` remain in `PortfolioStrategy`
  but `CLAUDE.md` documents them as current while the README does not).

Either wire them up or delete them. Printing an inert setting as though it's active is
worse than not having it.

### 2.7 `unwrap()`/`panic!` as the error strategy on all input paths

`ingest.rs` alone: `result.unwrap()` on every CSV record, `parse().unwrap()` on every
numeric field, `expect("file access error")`, and `get_team` panics on a missing
`(region, seed)`. A malformed bracket file or an off-format CSV aborts with a
backtrace rather than a message. Adopt `anyhow` at the boundary and `thiserror` for
the library types.

---

## Tier 3 — Performance

Current timings on 4 cores, `--source csv`:

| workload | wall |
|---|---|
| GA, 200 gens, pool 10k | 10.8 s |
| portfolio ga-whole, 5 brackets, 50 gens, pool 10k | 13.2 s |

Not slow, but the hot path has structural waste, and §1.1 makes most of it moot for
single brackets.

### 3.1 Shrink `FastBracket` to fit in cache

`FastBracket` is `[u8; 63]` + `[i32; 63]` = 315 bytes. A 10k pool is 3.1 MB — well past
L2, so every scoring pass streams from L3/RAM.

The seed array is redundant: seed is a function of the winner's team index. Drop it,
keep a 64-entry `seed_of[team_index]` lookup, and `FastBracket` becomes 63 bytes —
a 10k pool is 630 KB and fits in L2. Expect a solid multiple on scoring throughput,
which is where essentially all the time goes.

Further: for a given candidate, precompute `pts[i] = table[round(i)][seed_of[winner[i]]]`
once (63 values), so the inner loop over the pool is a compare-and-add with no
indirection.

### 3.2 Nested `par_iter` is fighting itself

`GeneticAlgorithm::evaluate_fitness` (`ga.rs:420`) does
`population.par_iter().map(|ind| pool.score_bracket(...))`, and `score_bracket`
(`ga.rs:51`) internally does `fast_brackets.par_iter()`. That's 100 outer rayon tasks
each spawning a 10,000-item parallel reduction. Rayon handles it safely, but the
scheduling overhead is pure loss when the outer level already saturates the cores.

**Fix:** parallelize at the population level only; make the inner pool loop a plain
sequential fold. Same for `WholePortfolioGA::evaluate_fitness` (`ga.rs:744`).

### 3.3 Best-ball rescoring is O(N × pool) per SA step when it could be O(pool)

`anneal.rs::optimize_portfolio` mutates one bracket and then calls
`score_portfolio_best_ball` on the whole portfolio — re-converting all N brackets to
`FastBracket` and re-scoring all of them against all 10,000 sims. At the default
10,000 steps × 5 brackets that's 500M scoring operations, ~80% of them recomputing
values that did not change.

**Fix:** cache, per simulation, the top-two portfolio scores. When bracket `i`
changes, recompute only bracket `i` against each sim; the new max is
`max(new_i, best_excluding_i)`, and `best_excluding_i` is recoverable from the cached
top-two. Drops the per-step cost to O(pool) — a 5× win at N=5, more at larger N, and
it makes the revert path free.

### 3.4 Season fetches are fully sequential and all-or-nothing

`fetch_games_for_range` walks ~158 days one HTTP request at a time, with a 250 ms
sleep between NCAA calls (≈40 s of pure sleep). The cache is a single season-wide blob
with a 6-hour staleness window, so **any** refresh re-fetches all 158 days.

Completed games are immutable. Cache per-day, refetch only days at or after the last
incomplete day, and bound concurrency to the documented rate limit (5 req/s) rather
than serializing.

### 3.5 `unsafe` used to skip bounds checks on data-derived indices

`ScoreTable::get` (`bracket.rs:63`) and `ProbabilityCache::get` (`ingest.rs`) both use
`get_unchecked` on a `seed` / `team_index` that originates in a JSON or CSV file. A
seed of 0 or 17 from a malformed bracket file is undefined behavior, not a panic.

These bounds checks are perfectly branch-predicted and cost close to nothing next to
the memory traffic. Remove the `unsafe`, or validate the ranges once at ingest and
encode the invariant in the type (`Seed(NonZeroU8)` capped at 16).

---

## Tier 4 — Engineering hygiene

### 4.1 Seven tests for 5,830 lines, none covering the optimizer

`elo.rs` has 2, `api.rs` has 1, `portfolio.rs` has a few. `ga.rs`'s test module is a
comment listing the tests that would exist. There is no test that a bracket is
internally consistent, none for the binary round-trip, none for scoring, none for the
mutation and crossover operators — which is exactly why §0.1 and §0.3 survive.

Highest-value tests, roughly in order:
- **Bracket legality invariant.** Every game's winner is one of its two participants,
  and each game's participants are the winners of its two feeding games. Assert this
  after `new`, `new_from_binary`, `mutate`, crossover, and constraint application.
  This single property test catches §0.1 immediately.
- **Encoding round-trip.** `binary → bracket → binary` is the identity for all 63 bits.
- **Operator postcondition.** After `force_team_to_round(t, r)`, team `t` actually
  appears as a winner in round `r`. Catches §0.3 immediately.
- **Scoring.** A bracket scored against itself equals its `score` field; disjoint
  brackets score 0.
- **Exact-EV cross-check** (once §1.1 lands): the DP's EV matches a large Monte Carlo
  estimate within its confidence interval. Validates both implementations at once.

### 4.2 No CI, no lint gate, no benchmarks

- `cargo build` emits **33 warnings**, including ~20 dead-code items. Add
  `cargo clippy -- -D warnings` and `cargo fmt --check` to CI, then clear the backlog.
- The codebase makes strong performance claims in comments ("was 18.9 billion calls
  per run", "called ~300 million times") with no benchmark backing them. Add
  `criterion` benches for bracket construction, scoring, and a fixed-budget GA run, so
  §3.1–§3.3 can be measured rather than asserted.

### 4.3 No reproducibility

Every RNG call is `rand::thread_rng()`, reached fresh inside hot loops (including
per-game in `Bracket::new` and per-bit in `random63bool`). There is no way to
reproduce a run, which makes the variance in §1.2 impossible to investigate and makes
optimizer regressions untestable.

Add `--seed <u64>`, thread a `StdRng` through the optimizers, and for the rayon paths
derive a per-chunk seed so results are deterministic *and* parallel. Print the seed
in the run banner.

### 4.4 Results are stdout-only

The output is a flat list of 63 team names. To actually enter a pool, a user
transcribes it by hand — and for a 5-bracket portfolio, five times. `EvolvingPool`
has an `export_to_file` that dumps raw `Vec<bool>`, which is not usable by a human.

Add `--output <path>` with `--format json|csv|html`. The HTML/SVG bracket rendering
is a couple hundred lines and turns this from a simulation into a tool someone
finishes a task with. JSON also makes the output diffable across runs and testable
in CI.

### 4.5 Dependency and repo cleanup

- **`pyo3`** is declared and never used anywhere in `src/`. It pulls in a Python build
  dependency and can fail the build on machines without a suitable interpreter. Remove.
- **`indicatif`**, **`fnv`** — declared, never imported. Remove.
- **`tokio` with `features = ["full"]`** — never imported directly; `reqwest`'s blocking
  client vendors what it needs. Remove the direct dependency.
- **`feature_branch_log.txt`** (393 KB) and **`.DS_Store`** are committed. Delete and
  extend `.gitignore` (currently two lines) with `.DS_Store`, `*.log`.
- `Cargo.toml` has no `[profile.release]` section. `lto = "thin"`, `codegen-units = 1`
  are free wins for a compute-bound binary.

### 4.6 `CLAUDE.md` documents strategies that no longer exist

It lists `--portfolio-strategy champion|diverse|annealing|ga` with default `ga`. The
actual clap enum is `champion|diverse|annealing|ga-whole|ga-sequential`, defaulting to
`ga-whole`. Every documented GA command in `CLAUDE.md` fails to parse. The README is
current; `CLAUDE.md` is not.

---

## Suggested sequencing

1. **§0.1–0.5** — stop producing wrong output. §0.1 and §0.2 are small and have clean
   reproductions.
2. **§4.1** (legality + round-trip + operator postcondition tests) — before any
   refactor, so the refactor is verifiable.
3. **§2.1** (positional encoding) — dissolves §0.3, unblocks §0.1's clean fix, and
   deletes code.
4. **§1.1** (exact EV + DP solver) — the biggest capability jump, and it also gives
   §4.1 a ground truth to test against.
5. **§1.4** (probability calibration) — everything downstream inherits this error.
6. **§1.3** (opponent model) — the change that makes the output decision-useful.
7. **§3.1–3.3**, then **§2.3–2.7**, **§4.2–4.6** as cleanup.


---

## Addendum: what shipped

Implemented in this branch:

| Item | Change |
|---|---|
| §0.1 | Constraints applied to the bit vector and rebuilt; `LockSet` repairs every optimizer candidate; result verified before return |
| §0.2 | Separate constrained path removed; hill climber compares against the incumbent's own score |
| §0.3 | `force_index_to_round` walks the team's real path; the seed-only and region-only heuristics are gone |
| §0.4 | `src/names.rs` — normalization, aliases, tiered matching, ambiguity as an error; field validation at ingest |
| §0.5 | Failed or partial fetches are hard errors; partial seasons are never cached |
| §1.1 | `src/exact.rs` + `src/advancement.rs`; `--optimization-mode exact` is the default; heuristics report their gap |
| §2.5 | One layered `ScoringConfig`: defaults <- YAML <- CLI |
| §2.1 | Partial: structure centralized in `src/tree.rs`, one encode/decode path |

Also fixed along the way:

- `Bracket::expected_value` was not an expected value (§1.1 addendum).
- `Team` equality compared names, so two teams sharing a name were equal. It now
  compares index.
- `pretty_print`'s headings named the round just played, so `--lock-team
  X:FinalFour` listed X under a heading reading "Elite 8" and looked ignored.
  Headings now name the round the listed teams advance to.
- Bad `--seed-r*` values warned and silently fell back to `none`; now an error.
- The `unsafe get_unchecked` in `ScoreTable::get` on a data-derived seed is gone
  (§3.5); seeds are validated at ingest instead.

Test count went from 7 to 54, covering bracket legality across every construction
path, encoding round-trip, the mutation operator's postcondition, lock durability
under mutation, exhaustive verification of the DP, and local optimality of its
result.

Still open: §1.2 (overfitting / hold-out validation), §1.3 (opponent model),
§1.4 (probability calibration), §1.5 (ELO gaps), §2.2-2.4, §2.6-2.7, §3.1-3.4,
§4.1-4.6.
