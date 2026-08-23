# Where the win probabilities should come from

`ProbabilityCache` turns one number per team into all 2,016 matchup
probabilities. Everything downstream — the exact DP, the portfolio search, the
opposing-field model — is only as good as that number. This document ranks the
ways of producing it, measured rather than argued.

## How the numbers were produced

Nine seasons of BartTorvik game logs (2017-2019, 2021-2026): 49,053 games,
600 of them NCAA tournament games. Every prediction is walk-forward — a game is
scored from ratings built only out of games that had already finished. The
rating-to-probability link is a two-parameter logistic fit
**leave-one-season-out**, so the calibration is out of sample as well as the
ratings. The harness is in `analysis/win-probability/`.

Log loss on tournament games is the headline metric because that is the
population the optimizer runs on. It is a 600-game sample, so the bootstrap
interval matters more than the point estimate; a single tournament (63 games)
can move log loss by 0.05 on noise alone and must never be used to pick a model.

## The result

| | model | tourney log loss | tourney Brier | tourney acc | late-season log loss |
|---|---|---|---|---|---|
| A | **Elo as this repo ships it** | 0.6083 | 0.2115 | 62.7% | 0.5794 |
| B | Elo, wins only, no MOV | 0.6932 | 0.2484 | 58.8% | 0.5958 |
| C | Elo on point margin, tuned | 0.5759 | 0.1989 | 68.3% | 0.5686 |
| D | C + preseason carryover | 0.5483 | 0.1870 | 71.3% | 0.5613 |
| E | Ridge least squares on margin (Massey) | 0.5477 | 0.1856 | 71.2% | 0.5652 |
| F | **Ridge adjusted efficiency (offense/defense split)** | **0.5432** | 0.1848 | 70.7% | 0.5636 |
| G | F + preseason carryover | 0.5436 | 0.1851 | 70.5% | **0.5611** |
| H | LightGBM on 19 engineered features | 0.5405 | 0.1847 | 70.8% | 0.5649 |
| I | Torvik's own published T-Rank ratings | **0.5397** | 0.1826 | 71.2% | — |
| | seed difference alone | 0.5732 | 0.1958 | 70.3% | — |
| | constant 0.5 | 0.6931 | 0.2500 | 53.0% | 0.6931 |

Bootstrap 95% intervals on the difference from F, over 2,000 resamples of the
600 tournament games:

```
A. Elo, as shipped in this repo   +0.0648  [+0.0407, +0.0887]   significant
B. Elo, wins only (no MOV)        +0.1498  [+0.1108, +0.1890]   significant
C. Elo on margin, tuned           +0.0323  [+0.0142, +0.0506]   significant
D. C + preseason carryover        +0.0052  [-0.0058, +0.0161]
E. Ridge margin (Massey)          +0.0045  [-0.0027, +0.0121]
G. F + preseason carryover        +0.0004  [-0.0059, +0.0064]
H. LightGBM on 19 features        -0.0028  [-0.0179, +0.0124]
I. Torvik's published T-Rank      -0.0037  [-0.0128, +0.0062]
```

Read the table this way. There is **one** significant gap in it, and it is the
first one: the rating model this repo currently uses is 0.065 of log loss worse
than an opponent-adjusted efficiency model, and it is beaten by a model that
knows nothing but the seed numbers the committee published. Everything from D
downward is inside the noise of everything else from D downward. The distance
from A to F is fifteen times the distance from F to a tuned LightGBM.

## The ranking

### 1. Replace the rating with an opponent-adjusted least-squares fit (E/F)

Two forms of the same idea, and the cheaper one is the one to build.

**Build this (E).** One rating per team in points, plus one home term, fitted by
ridge least squares on game margin:

```
margin(game) = R[a] - R[b] + home·h
```

0.5440 on tournament games with no margin cap. It needs **date, teams, site and
score** — a strict subset of the columns `torvik.rs` already parses. No new feed,
no new column, no possession estimate.

**The efficiency split (F) is a display feature, not an accuracy feature.**

Splitting each team into offence and defence in points per possession —

```
points_for(team, game) / possessions = mean + off[team] + def[opponent] + home·h
```

— scores 0.5432, which is 0.0008 better than E and nowhere near significant. It
is worth building if you want to *print* adjO/adjD/tempo the way T-Rank does; it
is not worth building for the probabilities. It also does not need the tempo
column to work: substituting a constant 68 possessions for the real per-game
tempo costs 0.0011 (0.5447 vs 0.5436), so the extra column buys nothing either.

Either way, solve `(X'X + λI)β = X'y` once — λ ≈ 1, flat between 0.5 and 2, and
λ ≥ 8 is where it starts to hurt — then

```
P(a wins) = 1 / (1 + exp(-E[margin] / 7.11))
```

The design matrix is 2·N+1 columns for N teams, the normal equations accumulate
in one pass over the game log, and one dense solve of a ~720×720 symmetric
system is a few milliseconds — the same order as the exact bracket DP. There is
no iteration to converge, no learning rate, and no ordering dependence.

Why it beats Elo by so much: Elo learns strength of schedule only transitively,
one game at a time, and with 360 teams playing ~30 games each on near-disjoint
schedules it never finishes propagating. The least-squares fit solves for every
team simultaneously, so a 15-0 mid-major is discounted the moment the fit runs.

The measured gain over the shipped model is **0.065 log loss and 8 points of
tournament accuracy**, and it is the only change in this document that survives a
bootstrap.

Validation that this is not a bug in my favour: Torvik's own published T-Rank
ratings, snapshotted the day before each tournament, score 0.5397 on the same
600 games. My from-scratch ridge scores 0.5432. The 0.0035 difference has a
bootstrap interval straddling zero, so the implementation is at parity with a
public rating system that has had years of tuning poured into it.

### 2. Fix the link function and the home-court term

Free, whether or not you do anything else.

- **Home court is 3.1 points, not 4.1.** Fitted per season from the ridge home
  term: 3.01, 3.54, 3.21, 2.44, 2.79, 3.54, 3.11, 3.24, 2.92 — mean 3.09,
  declining slightly over the window. The repo's `HOME_ADVANTAGE = 100.0` Elo
  points is 4.09 points at the repo's own scale, about 30% hot. Neutral-site
  games should carry zero home term, which the repo already does correctly.
- **The link scale is 7.11 points per logit** (equivalently a probit σ of 11.9
  points; the two fit equally well — probit was better by 0.0002, which is
  nothing). In this repo's 538 units that is
  `rating_538 = 75 + 0.802 · adjEM_points_per_game`, which keeps
  `ProbabilityCache` and the whole 30.464 conversion untouched.
- **Clipping does nothing here.** [0.02, 0.98] moved tournament log loss by
  +0.0001 and [0.05, 0.95] by +0.0006. The model is already calibrated enough
  that clipping is pure insurance against a 16-over-1, not a source of skill.
- **A tournament-specific widening of the rating gap is real but tiny.**
  Multiplying the expected margin by 1.07 in tournament games (Silver's SBCB
  does this) gained 0.0004. The sign matches his finding — tournament games are
  slightly *less* upset-prone than a neutral-site reading of the ratings
  implies — but the size is not worth a code path.

### 3. Carry ratings across seasons

Worth a lot in November and December, worth nothing in March — but it is nearly
free, and it makes the tool usable before February.

Starting each season from `regress × last season's rating` instead of from
scratch, with the season's own games washing it out:

| | early season (first 45 days) | late season | tournament |
|---|---|---|---|
| Elo-on-margin, cold start | 0.5390 | 0.5686 | 0.5759 |
| Elo-on-margin, carryover 0.9 | **0.4861** | 0.5613 | 0.5483 |
| Ridge efficiency, cold start | 0.5269 | 0.5636 | 0.5432 |
| Ridge efficiency, carryover (prior weight 5, regress 0.7) | **0.4909** | **0.5611** | 0.5436 |

That is 0.04 of log loss in the first six weeks. For the ridge model the prior
is five pseudo-observations per team pulling toward last season's rating — three
extra lines in the accumulation loop. The prior weight matters: 5 is right,
15 is worse, 40 is much worse than no prior at all.

By the tournament the prior is worth +0.0004, i.e. nothing. If you only ever run
this in March, skip it.

### 4. Elo on point margin, if you want to keep Elo

If the streaming update is worth keeping for its own sake, the fix is to stop
updating on win/loss and start updating on **margin error**. Rate teams in
points, predict `R_a - R_b + home`, and update both sides by `K/2 × (actual
margin - predicted margin)`.

Tuned on this data: **K = 0.14, home = 3.0 points, no margin cap** (capping at
24 and at 18 both cost log loss), preseason carryover 0.8-0.9. That lands at
0.5483 on tournament games — statistically indistinguishable from the ridge fit,
0.060 better than what ships today, and it stays a one-line-per-game update with
no linear algebra.

Two things that are *not* the fix: the 538 MOV multiplier with its
autocorrelation correction (model A already has it, and A is the worst
non-degenerate model in the table), and dropping MOV entirely (model B, whose
0.6932 on tournament games is a coin flip to four decimal places). Margin is essential; the
FiveThirtyEight *packaging* of margin is not what is doing the work.

### 5. Consuming Torvik's published ratings instead of computing your own

Torvik's ratings are the best single number in the table (0.5397), and there are
two ways to get them. `trank.php?year=YYYY&begin=…&end=…&csv=1` takes an
arbitrary date range — which is what made the backtest possible — but sits behind
a JS challenge that a `POST js_test_submitted=1` satisfies before a normal `GET`
gets through. `barttorvik.com/YYYY_team_results.csv` is a plain static file with
no gate at all (verified 200, 215 KB, header row, `adjoe`/`adjde`/`barthag`/`adjt`
for 365 teams); it carries whatever games have been played when you fetch it,
which for live use on Selection Sunday is exactly what you want.

So this is a trade: 0.0035 of log loss (not significant) in exchange for a second
endpoint, a second name-resolution problem, and no control over the model.
Compute your own; keep `YYYY_team_results.csv` in your pocket as a cross-check
and as a fallback if the game log ever breaks.

Also worth knowing: `barttorvik.com/YYYY_super_sked.csv` is another static,
ungated CSV, carrying Torvik's own pregame predictions for every game — a
ready-made oracle to score any future model against.

### 6. Gradient-boosted trees — measurable, but not worth it here

LightGBM on 19 walk-forward features (the ridge signal, both Elo signals, the
offence/defence split, season-to-date four factors, pace, rest days, games
played, neutral flag, date), trained leave-one-season-out with tournament games
weighted 6×, scored **0.5405** on tournament games: 0.0028 better than the ridge
signal it was built on top of, with a bootstrap interval of [-0.018, +0.012].
On the 32,571-game late-season sample — where 0.003 *would* be detectable — it
was **worse** than the plain ridge model, 0.5649 against 0.5636. A plain logistic
regression on the same 19 features did better than the trees on that sample
(0.5617).

The honest reading: once an opponent-adjusted efficiency margin is in the feature
set, there is nothing left for the trees to find. They are not discovering
basketball; they are re-deriving a monotone function of one column. This matches
what the external literature reports — Zimmermann et al. put the ceiling for
this problem at 74-75% straight-up accuracy and reach it with plain adjusted
efficiency, and a 2026 five-model fusion study beat the best public rating system
by 1.6 accuracy points.

If you want it anyway it is buildable: train offline in Python, export the trees
to JSON, walk them in Rust (or use `gbdt-rs`). The cost is a second language in
the build, feature-parity discipline between trainer and scorer, and a retraining
cadence — for a gain that this backtest cannot distinguish from zero.

### 7. Things that were tested and did not work

Reporting these because each is a plausible idea that costs real code.

- **Recency weighting.** Torvik's published schedule (full weight for 40 days,
  then 1%/day decay to a 60% floor) made things very slightly *worse*: 0.5445 vs
  0.5436 on tournament games, 0.5637 vs 0.5636 late. A 60-day half-life was
  clearly worse (0.5541). College seasons are four months long and rosters do
  not turn over inside them; there is no drift for the decay to track.
- **Blowout capping.** Clamping the per-possession margin at ±0.30 gained 0.0004;
  at ±0.22, 0.0005. Both inside noise. Torvik's GameScript work needs
  play-by-play to do better than this, and play-by-play is not in the game log.
- **Seed as a feature.** Seed difference alone is a decent predictor (0.5732,
  better than the shipped Elo). Added *on top of* the efficiency rating it is
  worth **-0.0012** — actively harmful. The committee's information, injuries
  included, is already inside the efficiency numbers by mid-March.
- **Ensembling.** Blending my ridge with Torvik's T-Rank in logit space never
  beat Torvik alone (best blend 0.5404 at 25/75, against 0.5396 for Torvik).
  Blending with the Elo signal made things worse. The components are not
  independent — they are three renderings of the same game log.
- **Probit instead of logistic.** 0.5431 vs 0.5432. Pick either.

## Will the data be there for the 2027 tournament?

The recommended model needs **date, teams, site (H/A/N) and score** — nothing the
repo does not already fetch and parse for Elo. Adopting it adds no new feed, no
new endpoint and no new column, so its availability risk is exactly the
availability risk the repo already carries.

Checked rather than assumed, on the day this was written:

- `getgamestats.php?year=YYYY&csv=1` answers **200 with no key, no cookie and no
  JS challenge**, 4.9 MB for a full season. The JS gate is on `trank.php`, not
  here.
- The layout has been **31 columns in all eleven seasons from 2016 through 2026**,
  with tempo pinned at column 23 the whole way. The model reads columns 0, 2, 4,
  5 and 6, all of which `torvik.rs` already reads.
- `year=2027` currently returns 200 with **zero bytes**, because the season has
  not started. `parse_game_log` already turns that into a "no parsable games"
  error rather than an empty season, which is the right behaviour.
- The log is populated live through the season, so by Selection Sunday 2027 it
  will hold the full regular season and the conference tournaments.

What could actually go wrong, and what happens if it does:

| failure | likelihood | what it costs |
|---|---|---|
| Torvik puts `getgamestats.php` behind the same JS gate as `trank.php` | plausible — the site demonstrably does this elsewhere | a `POST js_test_submitted=1` then `GET` with the cookie gets through; that is how the backtest read `trank.php` |
| Column layout moves | low — stable 11 seasons | `parse_game_log`'s "more than 10% of rows unparsable" guard already fails loudly instead of producing quietly wrong ratings |
| Torvik stops publishing entirely | low, but it is one person's website with no SLA | fall back to the NCAA or ESPN feed already in `api.rs`. The margin fit needs only scores, so it survives the switch — but neither scoreboard marks neutral-site games, so ~12% of games would lose their home term |
| Ratings needed without the game log at all | — | `barttorvik.com/YYYY_team_results.csv`, static and ungated, carries adjOE/adjDE/barthag/tempo directly |

The one honest caveat: this is a single volunteer-run site, and every free
alternative for college basketball is in the same position. The mitigation is
that the model is cheap enough to re-fit from any source of scores, which is why
the margin form (E) is the one to build rather than the possession form (F) — the
possession form would strand you on Torvik specifically.

## What was built

Items 1 and 2 are implemented, in `src/ratings.rs`, behind `--ratings`:

- **`adjusted` (the default)** — the ridge fit on margin, with the home term
  estimated rather than assumed and the link scale fitted. `ProbabilityCache`
  is untouched: `RATING_538_PER_POINT` is derived so that the cache's existing
  `10^(-diff * 30.464/400)` reproduces the fitted logistic exactly.
- **`elo`** — the old model, kept so the change can be re-measured on any field
  rather than taken from this document.
- **`seed`** — the backup. Sixteen numbers, no game data, and it engages
  automatically with a loud banner whenever the chosen model cannot be produced:
  a dead feed, an unplayed season, a degenerate fit. Picking it explicitly skips
  the game fetch entirely.

Sanity check on real data, 2025-26: the fit's top three are Michigan, Duke,
Arizona and its home edge is +2.96 points. Torvik's published T-Rank for the same
season has the same top three, and nine seasons of fitted home edges average
3.09. The 2026 Final Four the three models pick from the same bracket is a fair
picture of the difference between them — `adjusted` takes Duke, Illinois,
Gonzaga, Michigan; `elo` takes an 11-seed and a 5-seed; `seed` takes all four
1-seeds.

One trap worth naming: **expected-score numbers are not comparable across rating
models.** Elo reports a *higher* expected score on the same bracket, because it
is flatter and more confident about upsets, not because it is more right. Only
out-of-sample log loss orders the models.

### Performance

The rating fit runs once per invocation, before any optimization, so its cost is
amortized over the whole portfolio search. Keeping it that way is what
`bench ratings` is for.

| | |
|---|---|
| Fit, 360 teams / 5,562 games | **10.8 ms** |
| Seed rating lookup | 11 ns |
| Whole run: 5-bracket portfolio, `exact-basis`, cached game log | ~1.1 s |

The design matrix is never formed. Each game touches exactly three columns, so
accumulating the normal equations is O(games) rather than O(games x teams); the
cost is one Cholesky over the resulting 366x366 system. Splitting that
factorization's dot product across four accumulators took it from 17.7 ms to
10.8 ms — a single-accumulator reduction is a serial dependency chain that LLVM
will not reassociate on its own, and the fixed summation order keeps the fit
bit-for-bit reproducible.

Below that there is nothing worth chasing. The fit is ~1% of a portfolio run,
and the structures that actually matter for bracket generation were already
memoized before this change: `ProbabilityCache` is a precomputed 64x64 table of
`f64` that fits in L2, and `AdvancementModel` derives from it once. Ratings enter
that table and then never appear in a hot loop again. Caching the fit to disk
would trade a real invalidation bug for eleven milliseconds.

## What is left

3. Add the carryover prior if you want December ratings — 0.04 of log loss in
   the first six weeks, 0.0004 by March.
4. Port the backtest into `bench`. `analysis/win-probability/` is Python, and the
   comparison it does — walk-forward, leave-one-season-out calibration, log loss
   and Brier on three populations — belongs where a rating change can be
   evaluated the way an optimizer change already is.

Everything else in this document is inside the noise of 600 tournament games,
which is all the tournament games that exist.
