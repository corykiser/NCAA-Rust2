# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NCAA March Madness bracket optimizer written in Rust. Uses ELO ratings calculated from live game data (ESPN/NCAA APIs) or historical FiveThirtyEight CSV data.

The single-bracket problem is solved **exactly** (`src/exact.rs`) — maximizing expected score is a dynamic program over the bracket tree, not a search. Genetic algorithms and simulated annealing remain for multi-bracket portfolios, where best-ball scoring takes a maximum over brackets and has no closed form.

## Build and Run Commands

```bash
# Build the project
cargo build --release

# Run with default settings (ESPN data source)
cargo run --release

# Run with specific data source
cargo run --release -- --source espn    # ESPN API (default)
cargo run --release -- --source ncaa    # NCAA API
cargo run --release -- --source csv     # FiveThirtyEight CSV

# Generate portfolio of diverse brackets
cargo run --release -- --portfolio 5 --portfolio-strategy ga-whole
cargo run --release -- --portfolio 5 --portfolio-strategy ga-sequential
cargo run --release -- --portfolio 5 --portfolio-strategy annealing
cargo run --release -- --portfolio 5 --portfolio-strategy champion
cargo run --release -- --portfolio 5 --portfolio-strategy diverse

# ELO-only mode (skip bracket optimization)
cargo run --release -- --elo-only

# Lock specific teams to reach rounds
cargo run --release -- --lock-team "Duke:FinalFour" --lock-team "UConn:Winner"

# Run tests
cargo test

# Run single test
cargo test test_expected_score
```

## Architecture

### Core Data Flow
1. **Data Ingestion** (`api.rs`, `ingest.rs`): Fetch game results from ESPN/NCAA APIs or CSV, parse into `GameResult` structs
2. **ELO Calculation** (`elo.rs`): Process games chronologically to calculate team ratings with margin-of-victory adjustments
3. **Tournament Setup** (`ingest.rs`): Create `TournamentInfo` with 64 teams organized by region and seed
4. **Bracket Simulation** (`bracket.rs`): Generate brackets using Monte Carlo simulation with probability-weighted outcomes
5. **Optimization** (`pool.rs`, `anneal.rs`): Evolve brackets using genetic algorithms or simulated annealing to maximize expected value
6. **Portfolio Generation** (`portfolio.rs`): Create diverse bracket portfolios with different champion strategies

### Key Structs

- **`TournamentInfo`**: Holds 64 teams with `RcTeam` (Arc<Team>) for efficient sharing, plus the derived tables the optimizers need (`seed_of`, `region_rank`, `r1_teams`, `r1_game_of_team`, `advancement`). The field is validated on construction — duplicate `(region, seed)` slots, duplicate names, and out-of-range seeds are rejected rather than silently overwriting each other.
- **`Bracket`**: 63 games stored as flat vector (R1: 0-31, R2: 32-47, R3: 48-55, R4: 56-59, R5: 60-61, R6: 62), uses binary representation for mutations
- **`Game`**: Single matchup with win probabilities calculated via logistic function: `1 / (1 + 10^(-rating_diff * 30.464 / 400))`
- **`ScoringConfig`**: Configurable round scoring with seed bonuses (Add, Multiply, or None per round)

### Parallel Processing
Uses `rayon` for parallel bracket generation and scoring. Teams use `Arc<Team>` for thread-safe reference counting without data cloning.

## CLI Arguments Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | espn | Data source: espn, ncaa, csv |
| `--season` | auto | Season format: "2024-2025" |
| `--tournament-year` | - | Year to fetch bracket teams |
| `--bracket-file` | - | Local JSON file with bracket teams |
| `--generations` | 200 | Genetic algorithm generations |
| `--batch-size` | 1000 | Monte Carlo simulations per scoring (legacy mode) |
| `--portfolio` | - | Number of brackets to generate |
| `--portfolio-strategy` | ga-whole | Strategy: ga-whole, ga-sequential, annealing, champion, diverse |
| `--diversity-weight` | 5.0 | Weight for bracket diversity (legacy strategies) |
| `--lock-team` | - | Lock team to round (repeatable) |
| `--score-r1` through `--score-r6` | 1,2,4,8,16,32 | Points per round |
| `--seed-r1` through `--seed-r6` | add/multiply | Seed scoring mode per round |
| `--config` | - | Path to YAML configuration file |
| `--generate-config` | - | Print sample config.yaml and exit |
| `--optimization-mode` | exact | Single bracket mode: exact, ga, hybrid, legacy |
| `--allow-partial-data` | false | Proceed despite missing/incomplete game data |
| `--population-size` | - | Override GA population size |
| `--pool-size` | - | Override simulation pool size |
| `--smart-mutation` | - | Override smart mutation setting |
| `-v, --verbose` | false | Verbose output during optimization |

## Configuration File

Settings can be configured via `config.yaml` (auto-loaded if present):

```yaml
scoring:
  round_scores: [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
  seed_scoring: ["add", "add", "add", "multiply", "multiply", "multiply"]

ga:
  population_size: 100
  generations: 200
  mutation_rate: 0.15
  crossover_rate: 0.7
  elitism_count: 5
  tournament_size: 3
  smart_mutation: true
  smart_mutation_rate: 0.8

simulation:
  pool_size: 10000

portfolio:
  num_brackets: 5
  best_ball: true
```

Generate a sample config: `cargo run -- --generate-config > config.yaml`

## Genetic Algorithm (New)

The GA implementation (`src/ga.rs`) provides:

- **SimulationPool**: Pre-generates N random brackets for consistent scoring. Avoids regenerating random brackets each evaluation.
- **Best-Ball Scoring**: Portfolio fitness = average of max(scores) across simulations. Only one bracket needs to score well per simulation.
- **Smart Mutation**: Instead of random bit flips, picks a team and round, forces that team to advance to that round. Changes propagate to earlier rounds.
- **Sequential Portfolio Optimization**: Optimizes bracket 1, freezes it. Optimizes bracket 2 for marginal contribution to portfolio, freezes. Repeat.

Key commands:
```bash
# Single bracket with GA (default)
cargo run --release

# Portfolio with GA and best-ball scoring
cargo run --release -- --portfolio 5 --portfolio-strategy ga -v

# Hybrid SA+GA for single bracket
cargo run --release -- --optimization-mode hybrid -v

# Use legacy hill-climbing
cargo run --release -- --optimization-mode legacy
```

## Exact Single-Bracket Solver (`src/exact.rs`)

Expected score is `SUM over games of points(round, seed of pick) * P(pick wins that game)`.
The advancement probabilities (`src/advancement.rs`) depend only on the rating
model, so the objective is linear and the only coupling between games is
structural. That makes it a DP over the bracket tree:

```text
best[game][team] = points(round, seed) * P(team wins game)
                 + best[child containing team][team]
                 + MAX over u of best[other child][u]
```

64 teams over 6 rounds. Provably optimal, deterministic, ~3 ms including data
loading — against ~11 s and a different champion each run for the GA.

`--lock-team` drops into the same recurrence: forcing a team to win a game just
removes every other team from that game's candidate set. Conflicting locks are
detected before any work happens.

The heuristic modes print their gap to the exact optimum, which is what makes
their output interpretable.

## Bracket Structure (`src/tree.rs`)

The flat 63-game layout lives in one place. Within a region the eight round-1
games are in **seed order** (`1v16, 2v15, 3v14, ...`), which is not tree order —
the `1v16` winner plays the `8v9` winner. `CHILDREN`, `PARENT`, and `ROUND_OF`
encode this; nothing else should re-derive it.

## Binary Encoding

Each bracket is 63 booleans. A bit is `true` when the numerically lower seed
wins, or — for the cross-region Final Four and final — when the alphabetically
earlier region wins. The definition lives in `TournamentInfo::bit_true_winner`,
and `decode_winners` / `binary_from_winners` are the only encode/decode paths.

A bit's meaning depends on who reaches that game, so flipping an early bit
changes what later bits refer to. `TeamRoundMutator::force_index_to_round` is
the directed alternative: it walks the team's actual path up the parent chain
and sets each bit against the real opponent.

## Team Locks (`LockSet`)

Mutation and crossover rewrite the bit vector, so a lock applied only to a
starting bracket is gone after one generation. `LockSet::repair` is applied to
every candidate the optimizers produce, and `ConstrainedBracketBuilder` verifies
the constraints hold before returning.

## Configuration Precedence

Built-in defaults, then `config.yaml`, then CLI flags. The `--score-r*` and
`--seed-r*` flags are `Option`s so "not passed" is distinguishable from "passed
the default", which is what lets the config file's `scoring:` section take
effect.

## Testing

`cargo test`. The suite is property-oriented:

- **Legality**: every bracket from every path — random, decoded, mutated,
  crossed over, constrained, exact — satisfies "the winner of a game played in
  it, and a game's participants are the winners of its two feeding games".
- **Round-trip**: `binary -> winners -> binary` is the identity.
- **Operator postcondition**: after `force_index_to_round(t, n)`, team `t`
  actually wins `n` games, for all 64 teams at all 6 depths.
- **Lock durability**: locks still hold after 200 rounds of mutation.
- **Exhaustive**: with the last two rounds scored at zero the regions decouple,
  so the DP's answer is checked against brute force over all 2^15 outcomes of
  each region.
- **Local optimality**: no single bit flip and no forced team-round move beats
  the exact solution.
- **Calibration**: the exact expected value agrees with a 40,000-trial Monte
  Carlo estimate.

## Data Caching

API responses are cached in `./data/` directory:
- `games_{season}.json`: Season game results (6-hour staleness)
- `bracket_{year}.json`: Tournament bracket teams
