# NCAA March Madness Bracket Optimizer

A bracket optimization tool built in Rust. It solves the single-bracket problem
exactly, and uses genetic algorithms and Monte Carlo simulation for multi-bracket
portfolios.

## Features

- **Exact single-bracket solver**: the highest-expected-value bracket, proven optimal, in milliseconds
- **Multiple Data Sources**: ESPN API, NCAA API, or FiveThirtyEight CSV files
- **Genetic Algorithm Optimization**: Population-based evolution with Team-Round mutation and crossover
- **Portfolio Optimization**: Generate diverse bracket portfolios optimized for best-ball scoring
- **Monte Carlo Simulation**: Score brackets against thousands of simulated tournament outcomes
- **Configurable Scoring**: Support for various scoring systems (per-round points, seed bonuses)
- **Team locks**: pin any team to any round, honored exactly and preserved through optimization

## How It Works

### Core Concepts

**Exact expected value**: A bracket's expected score is the sum over its 63
picks of `points × P(that pick actually wins that game)`. The advancement
probabilities depend only on the rating model, never on your picks, so they are
computed once by a pass over the bracket tree. No sampling, no error bars.

**Monte Carlo Scenarios**: The optimizer generates thousands of random tournament brackets based on team win probabilities (derived from ELO ratings). These scenarios represent possible tournament outcomes.

**Best-Ball Scoring**: For a portfolio of N brackets, the score against each scenario is the *maximum* score among all brackets. The fitness is the average best-ball score across all scenarios. This naturally encourages diversity.

**Team-Round Pairs**: The genetic algorithm operates on Team-Round pairs as genes (e.g., "Duke reaches the Final Four"). This is semantically meaningful - mutations and crossovers preserve bracket consistency.

### Single-Bracket Modes (`--optimization-mode`)

The expected-value objective is linear in the advancement probabilities, and the
only coupling between games is structural — the team you pick to win a game must
be a team you already advanced into it. That makes it a dynamic program over the
bracket tree rather than a search problem:

```text
best[game][team] = points(round, seed) × P(team wins game)
                 + best[child containing team][team]
                 + MAX over u of best[other child][u]
```

64 teams over 6 rounds, a few thousand multiply-adds.

| Mode | What it does |
|---|---|
| `exact` (default) | Dynamic program. Provably optimal, deterministic, ~3 ms. |
| `ga` | Population-based GA against Monte Carlo scenarios. |
| `hybrid` | Simulated annealing with GA operators. |
| `legacy` | Hill climbing. |

The heuristic modes print their gap to the exact optimum, so their output can be
read against the number it should be compared with:

```text
Expected score of this bracket: 232.88
Exact optimum:                  233.38  (gap 0.50, 99.8% of optimum)
```

Monte Carlo is still what portfolio mode needs: best-ball scoring takes a
maximum over your brackets, which is not linear and has no equivalent closed
form.

### Portfolio Modes (`--portfolio-strategy`)

1. **Exact basis + coordinate ascent** (`--portfolio-strategy exact-basis`, default)
   - Builds every bracket that is exactly optimal subject to one extra
     advancement requirement — 384 constrained solves, a few milliseconds — and
     greedily selects the entries that add most to best-ball
   - Then sweeps entries one at a time, replacing each with the best alternative
     the neighbourhood offers, until nothing improves
   - Deterministic, and the strongest of these by held-out best-ball score
   - Also reports a held-out score on an independently drawn scenario pool, so
     you can see how much of the gain is real rather than fitted to the sample

2. **Whole Portfolio GA** (`--portfolio-strategy ga-whole`)
   - Evolves entire portfolios as individuals
   - Each portfolio contains N brackets
   - Fitness = best-ball score against Monte Carlo scenarios

3. **Sequential Portfolio** (`--portfolio-strategy ga-sequential`)
   - Optimizes brackets one at a time
   - Each new bracket is optimized for marginal contribution to the frozen portfolio
   - Often finds better solutions than whole portfolio evolution

4. **Simulated Annealing** (`--portfolio-strategy annealing`)
   - Classic SA optimization on portfolio
   - Uses Team-Round mutations

## Installation

```bash
git clone https://github.com/corykiser/NCAA-Rust2.git
cd NCAA-Rust2
cargo build --release
```

## Usage

### Basic Usage (ESPN API)

```bash
# Generate a 5-bracket portfolio (exact basis + coordinate ascent, the default)
cargo run --release -- --portfolio 5 --portfolio-strategy ga-whole --generations 200 --pool-size 10000

# Generate using sequential optimization
cargo run --release -- --portfolio 5 --portfolio-strategy ga-sequential --generations 200 --pool-size 10000
```

### Using CSV Data

```bash
cargo run --release -- --source csv --csv-path fivethirtyeight_ncaa_forecasts.csv --portfolio 5
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--optimization-mode` | `exact`, `ga`, `hybrid`, or `legacy` (single bracket) | `exact` |
| `--portfolio N` | Number of brackets to generate | - |
| `--portfolio-strategy` | `ga-whole`, `ga-sequential`, `annealing`, `champion`, `diverse` | `ga-whole` |
| `--lock-team` | Pin a team to a round, e.g. `"Duke:FinalFour"` (repeatable) | - |
| `--generations N` | GA generations per optimization | 200 |
| `--pool-size N` | Number of Monte Carlo scenarios | 10000 |
| `--source` | Data source: `espn`, `ncaa`, or `csv` | `espn` |
| `--allow-partial-data` | Proceed despite missing game data (see below) | false |
| `--verbose` | Show detailed progress | false |

### Team Locks

```bash
# Solve exactly for the best bracket in which Duke reaches the Final Four
cargo run --release -- --lock-team "Duke:FinalFour"

# Locks compose, and everything they do not pin is still optimized
cargo run --release -- --lock-team "Duke:FinalFour" --lock-team "UConn:Winner"
```

Rounds are `Round2`, `Sweet16`, `Elite8`, `FinalFour`, `Championship`, `Winner`.
Locks that cannot hold together are rejected up front:

```text
Error: Houston and Alabama cannot both win their round-5 game — they meet there
```

Team names must identify exactly one team. An ambiguous or unknown name is an
error with suggestions, never a silent guess at a different school.

### Data Integrity

Ratings computed from a missing or partial season look exactly like real ones —
every team sits at the 1500 default and every matchup becomes a coin flip. A
failed or incomplete fetch is therefore a hard error, and an incomplete season
is never written to the cache. `--allow-partial-data` opts out.

### Scoring Configuration

```bash
# Custom scoring (points per round)
cargo run --release -- --portfolio 5 \
  --score-r1 1 --score-r2 2 --score-r3 4 \
  --score-r4 8 --score-r5 16 --score-r6 32 \
  --seed-r1 add --seed-r4 multiply
```

Seed scoring modes:
- `add`: Base points + seed number
- `multiply`: Base points × seed number
- `none`: Base points only

## Configuration File

Create a `config.yaml` for persistent settings:

```yaml
scoring:
  round_scores: [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
  seed_scoring: [Add, Add, Add, Multiply, Multiply, Multiply]

ga:
  population_size: 100
  generations: 200
  mutation_rate: 0.3
  crossover_rate: 0.8
  elitism_count: 5
  tournament_size: 3

simulation:
  pool_size: 10000
```

## Architecture

```
src/
├── main.rs          # CLI and orchestration
├── tree.rs          # Static bracket structure: child/parent/round tables
├── advancement.rs   # Exact per-game advancement probabilities
├── exact.rs         # Exact optimal bracket (dynamic program) and EV scoring
├── bracket.rs       # Bracket representation and scoring
├── picks.rs         # Compact bracket the optimizers work on: 63 winners + 63 bits
├── score.rs         # Scenario pool and the branchless SIMD scoring kernels
├── optimize.rs      # Portfolio construction: exact basis, greedy, coordinate ascent
├── ga.rs            # Genetic algorithm (MonteCarloScenarios, TeamRoundMutator, LockSet)
├── ingest.rs        # Data loading, field validation, team ratings
├── names.rs         # Deterministic team-name resolution
├── elo.rs           # ELO rating calculations
├── api.rs           # ESPN/NCAA API clients
├── portfolio.rs     # Portfolio management and constrained brackets
├── anneal.rs        # Simulated annealing optimizer
└── config.rs        # YAML configuration
```

The bracket layout lives in exactly one place, `tree.rs`. Within a region the
eight round-1 games are in seed order (`1v16, 2v15, ...`), which is *not* tree
order — the `1v16` winner plays the `8v9` winner. Every module that needs to
walk the bracket uses the child/parent tables rather than re-deriving that.

## Configuration Precedence

Built-in defaults, then `config.yaml`, then CLI flags. A flag you do not pass
leaves the config file's value in place; a flag you do pass wins.

## Performance

Single bracket, exact mode: ~3 ms including CSV parsing, and deterministic.

Portfolio mode is where the work goes, and it is dominated by one kernel:
scoring a candidate bracket against a pool of sampled tournaments. Three changes
account for most of the speedup over the original implementation.

**The scoring loop is branchless.** Written the obvious way — `if the winners
match { score += points }` — it compiles to a branch per game, and whether a
pick matches a sampled tournament is close to a coin flip, so one scenario costs
sixty branch mispredictions. `score.rs` compares sixteen games at a time with
SSE2 and masks the payouts instead.

**The scenario pool is bytes.** A scenario is 64 bytes of winner indices, not a
`Bracket` with three reference-counted team handles per game. A 10,000-scenario
pool went from ~75 MB and two million atomic refcount operations to 640 KB.

**Frozen work is not recomputed.** When a portfolio is fixed except for one
entry, the rest of it has a constant per-scenario best; evaluating a replacement
costs one bracket's scoring rather than the whole portfolio's.

Measured on four cores against the shipped 2023 field (`cargo run --release
--bin bench`):

| Benchmark | Before | After | |
|---|---:|---:|---:|
| Build a 10,000-scenario pool | 71.2 ms | 2.1 ms | 33x |
| Score one bracket vs 10k scenarios | 854 µs | 77 µs | 11x |
| Best-ball, 5 brackets vs 10k | 3.70 ms | 0.72 ms | 5.1x |
| Marginal contribution vs 10k | 4.32 ms | 0.13 ms | 34x |
| Force a team to a round | 354 ns | 23 ns | 15x |
| GA, single bracket, 100 pop x 20 gen | 1.63 s | 55 ms | 30x |
| GA, 5-bracket portfolio, 50 pop x 10 gen | 1.97 s | 86 ms | 23x |

Those are the same operations doing the same thing — `cargo test` checks the new
kernel against the original scorer pick for pick.

### Does it find better portfolios?

Speed only matters if the answers improve. Optimizing against a finite sample of
tournaments will always flatter itself, so the comparison below scores each
result on a *held-out* pool of 200,000 scenarios it was never optimized
against. Training pool 50,000 scenarios; GAs at 100 population by 200
generations.

| Entries | `exact-basis` | `ga-whole` | `ga-sequential` |
|---:|---:|---:|---:|
| 1 | **233.31** (0.1 s) | 231.94 (3.4 s) | 232.27 (2.4 s) |
| 3 | **276.70** (2.8 s) | 276.56 (8.7 s) | 274.52 (7.4 s) |
| 5 | **296.56** (5.0 s) | 295.34 (13.9 s) | 292.76 (12.4 s) |
| 10 | **323.23** (10.2 s) | 319.93 (26.7 s) | 321.26 (24.8 s) |

Reproduce with `cargo run --release --bin bench -- quality`.

The single-entry row is the tell: `exact-basis` returns the provably optimal
bracket there, and the GAs do not reach it even with 200 generations.

## Tests

```bash
cargo test
```

The suite checks the properties that matter rather than only unit behavior: that
every bracket produced by any path is structurally legal, that the bit encoding
round-trips, that forcing a team to a round actually gets it there, that locks
survive mutation, and that the dynamic program's answer matches exhaustive
enumeration of all 32,768 outcomes of each region.

## License

MIT License

Copyright (c) 2022-2025 Cory Kiser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
