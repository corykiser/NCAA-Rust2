# NCAA March Madness Bracket Optimizer

A high-performance bracket optimization tool built in Rust that uses genetic algorithms and Monte Carlo simulation to generate optimal bracket portfolios for March Madness pools.

## Features

- **Multiple Data Sources**: ESPN API, NCAA API, or FiveThirtyEight CSV files
- **Genetic Algorithm Optimization**: Population-based evolution with Team-Round mutation and crossover
- **Portfolio Optimization**: Generate diverse bracket portfolios optimized for best-ball scoring
- **Monte Carlo Simulation**: Score brackets against thousands of simulated tournament outcomes
- **Configurable Scoring**: Support for various scoring systems (per-round points, seed bonuses)

## How It Works

### Core Concepts

**Monte Carlo Scenarios**: The optimizer generates thousands of random tournament brackets based on team win probabilities (derived from ELO ratings). These scenarios represent possible tournament outcomes.

**Best-Ball Scoring**: For a portfolio of N brackets, the score against each scenario is the *maximum* score among all brackets. The fitness is the average best-ball score across all scenarios. This naturally encourages diversity.

**Team-Round Pairs**: The genetic algorithm operates on Team-Round pairs as genes (e.g., "Duke reaches the Final Four"). This is semantically meaningful - mutations and crossovers preserve bracket consistency.

### Optimization Modes

1. **Whole Portfolio GA** (`--portfolio-strategy ga-whole`)
   - Evolves entire portfolios as individuals
   - Each portfolio contains N brackets
   - Fitness = best-ball score against Monte Carlo scenarios

2. **Sequential Portfolio** (`--portfolio-strategy ga-sequential`)
   - Optimizes brackets one at a time
   - Each new bracket is optimized for marginal contribution to the frozen portfolio
   - Often finds better solutions than whole portfolio evolution

3. **Simulated Annealing** (`--portfolio-strategy annealing`)
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
# Generate a 5-bracket portfolio using whole portfolio GA
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
| `--portfolio N` | Number of brackets to generate | - |
| `--portfolio-strategy` | `ga-whole`, `ga-sequential`, or `annealing` | `ga-whole` |
| `--generations N` | GA generations per optimization | 200 |
| `--pool-size N` | Number of Monte Carlo scenarios | 10000 |
| `--source` | Data source: `espn`, `ncaa`, or `csv` | `espn` |
| `--verbose` | Show detailed progress | false |

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
├── bracket.rs       # Bracket representation and scoring
├── ga.rs            # Genetic algorithm (MonteCarloScenarios, TeamRoundMutator, etc.)
├── ingest.rs        # Data loading and team ratings
├── elo.rs           # ELO rating calculations
├── api.rs           # ESPN/NCAA API clients
├── portfolio.rs     # Portfolio management
├── anneal.rs        # Simulated annealing optimizer
└── config.rs        # YAML configuration
```

## Performance

The optimizer is highly parallelized using Rayon:
- Monte Carlo scenario generation
- Fitness evaluation across population
- Bracket scoring against scenarios

Typical performance (M1 Mac):
- 10k scenarios, 200 generations, 5 brackets: ~15-50 seconds
- 100k scenarios, 500 generations, 5 brackets: ~3-5 minutes

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
