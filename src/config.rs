// Configuration module for NCAA Bracket Optimizer
// Supports YAML configuration files for scoring, GA parameters, and simulation settings

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::bracket::{ScoringConfig, SeedScoring};

/// Root configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub scoring: ScoringSettings,
    #[serde(default)]
    pub ga: GaSettings,
    #[serde(default)]
    pub simulation: SimulationSettings,
    #[serde(default)]
    pub portfolio: PortfolioSettings,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            scoring: ScoringSettings::default(),
            ga: GaSettings::default(),
            simulation: SimulationSettings::default(),
            portfolio: PortfolioSettings::default(),
        }
    }
}

impl Config {
    /// Load configuration from a YAML file
    pub fn from_file(path: &str) -> Result<Self, String> {
        if !Path::new(path).exists() {
            return Err(format!("Config file not found: {}", path));
        }

        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse config file: {}", e))
    }

    /// Load configuration from file if it exists, otherwise use defaults
    pub fn load_or_default(path: Option<&str>) -> Self {
        match path {
            Some(p) => Self::from_file(p).unwrap_or_else(|e| {
                eprintln!("Warning: {}", e);
                eprintln!("Using default configuration.");
                Self::default()
            }),
            None => {
                // Try default locations
                for default_path in &["config.yaml", "config.yml", ".ncaa-config.yaml"] {
                    if Path::new(default_path).exists() {
                        if let Ok(config) = Self::from_file(default_path) {
                            println!("Loaded configuration from {}", default_path);
                            return config;
                        }
                    }
                }
                Self::default()
            }
        }
    }

    /// Save configuration to a YAML file
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let yaml = serde_yaml::to_string(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(path, yaml)
            .map_err(|e| format!("Failed to write config file: {}", e))?;

        Ok(())
    }

    /// Convert scoring settings to ScoringConfig
    pub fn to_scoring_config(&self) -> ScoringConfig {
        ScoringConfig {
            round_scores: self.scoring.round_scores,
            round_seed_scoring: [
                parse_seed_scoring(&self.scoring.seed_scoring[0]),
                parse_seed_scoring(&self.scoring.seed_scoring[1]),
                parse_seed_scoring(&self.scoring.seed_scoring[2]),
                parse_seed_scoring(&self.scoring.seed_scoring[3]),
                parse_seed_scoring(&self.scoring.seed_scoring[4]),
                parse_seed_scoring(&self.scoring.seed_scoring[5]),
            ],
        }
    }
}

/// Scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringSettings {
    /// Points awarded for correct pick in each round [R1, R2, S16, E8, F4, Championship]
    #[serde(default = "default_round_scores")]
    pub round_scores: [f64; 6],

    /// Seed scoring mode for each round: "add", "multiply", or "none"
    #[serde(default = "default_seed_scoring")]
    pub seed_scoring: [String; 6],
}

impl Default for ScoringSettings {
    fn default() -> Self {
        ScoringSettings {
            round_scores: default_round_scores(),
            seed_scoring: default_seed_scoring(),
        }
    }
}

fn default_round_scores() -> [f64; 6] {
    [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
}

fn default_seed_scoring() -> [String; 6] {
    [
        "add".to_string(),
        "add".to_string(),
        "add".to_string(),
        "multiply".to_string(),
        "multiply".to_string(),
        "multiply".to_string(),
    ]
}

/// Genetic Algorithm settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaSettings {
    /// Number of individuals in the population
    #[serde(default = "default_population_size")]
    pub population_size: usize,

    /// Number of generations to evolve
    #[serde(default = "default_generations")]
    pub generations: usize,

    /// Probability of mutation per individual (0.0 - 1.0)
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,

    /// Probability of crossover between parents (0.0 - 1.0)
    #[serde(default = "default_crossover_rate")]
    pub crossover_rate: f64,

    /// Number of top individuals to preserve each generation
    #[serde(default = "default_elitism_count")]
    pub elitism_count: usize,

    /// Number of individuals in tournament selection
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,

    /// Use smart mutation (team/round based) vs random bit flip
    #[serde(default = "default_smart_mutation")]
    pub smart_mutation: bool,

    /// Probability of using smart mutation vs bit flip (when smart_mutation is true)
    #[serde(default = "default_smart_mutation_rate")]
    pub smart_mutation_rate: f64,
}

impl Default for GaSettings {
    fn default() -> Self {
        GaSettings {
            population_size: default_population_size(),
            generations: default_generations(),
            mutation_rate: default_mutation_rate(),
            crossover_rate: default_crossover_rate(),
            elitism_count: default_elitism_count(),
            tournament_size: default_tournament_size(),
            smart_mutation: default_smart_mutation(),
            smart_mutation_rate: default_smart_mutation_rate(),
        }
    }
}

fn default_population_size() -> usize { 100 }
fn default_generations() -> usize { 200 }
fn default_mutation_rate() -> f64 { 0.15 }
fn default_crossover_rate() -> f64 { 0.7 }
fn default_elitism_count() -> usize { 5 }
fn default_tournament_size() -> usize { 3 }
fn default_smart_mutation() -> bool { true }
fn default_smart_mutation_rate() -> f64 { 0.8 }

/// Simulation pool settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSettings {
    /// Number of random brackets in the simulation pool
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,

    /// Whether to regenerate pool each generation (false = reuse)
    #[serde(default = "default_regenerate_pool")]
    pub regenerate_pool: bool,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        SimulationSettings {
            pool_size: default_pool_size(),
            regenerate_pool: default_regenerate_pool(),
        }
    }
}

fn default_pool_size() -> usize { 10000 }
fn default_regenerate_pool() -> bool { false }

/// Portfolio optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSettings {
    /// Number of brackets in the portfolio
    #[serde(default = "default_num_brackets")]
    pub num_brackets: usize,

    /// Use best-ball scoring (max score across portfolio per simulation)
    #[serde(default = "default_best_ball")]
    pub best_ball: bool,
}

impl Default for PortfolioSettings {
    fn default() -> Self {
        PortfolioSettings {
            num_brackets: default_num_brackets(),
            best_ball: default_best_ball(),
        }
    }
}

fn default_num_brackets() -> usize { 5 }
fn default_best_ball() -> bool { true }

/// Parse seed scoring string to enum
fn parse_seed_scoring(s: &str) -> SeedScoring {
    match s.to_lowercase().as_str() {
        "add" => SeedScoring::Add,
        "multiply" | "mult" => SeedScoring::Multiply,
        "none" | "off" => SeedScoring::None,
        _ => {
            eprintln!("Warning: Unknown seed scoring mode '{}', defaulting to None", s);
            SeedScoring::None
        }
    }
}

/// Generate a sample configuration file
pub fn generate_sample_config() -> String {
    r#"# NCAA Bracket Optimizer Configuration
# All values shown are defaults - uncomment and modify as needed

# Scoring configuration
scoring:
  # Points per round [R1, R2, Sweet16, Elite8, Final4, Championship]
  round_scores: [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
  # Seed scoring mode per round: "add", "multiply", or "none"
  # "add" = base_points + seed
  # "multiply" = base_points * seed
  # "none" = base_points only
  seed_scoring: ["add", "add", "add", "multiply", "multiply", "multiply"]

# Genetic Algorithm settings
ga:
  # Population size (number of brackets evolved simultaneously)
  population_size: 100
  # Number of generations to evolve
  generations: 200
  # Mutation rate (probability of mutating each individual)
  mutation_rate: 0.15
  # Crossover rate (probability of crossover between parents)
  crossover_rate: 0.7
  # Elitism (number of top individuals preserved each generation)
  elitism_count: 5
  # Tournament selection size
  tournament_size: 3
  # Use smart mutation (team/round based) instead of random bit flips
  smart_mutation: true
  # Probability of smart mutation vs bit flip when smart_mutation is enabled
  smart_mutation_rate: 0.8

# Simulation pool settings
simulation:
  # Number of random brackets for scoring (higher = more accurate but slower)
  pool_size: 10000
  # Regenerate pool each generation (false = reuse for consistency)
  regenerate_pool: false

# Portfolio settings
portfolio:
  # Number of brackets in portfolio
  num_brackets: 5
  # Use best-ball scoring (max score in portfolio per simulation)
  best_ball: true
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.ga.population_size, 100);
        assert_eq!(config.simulation.pool_size, 10000);
    }

    #[test]
    fn test_parse_yaml() {
        let yaml = r#"
ga:
  population_size: 200
  generations: 500
simulation:
  pool_size: 50000
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.ga.population_size, 200);
        assert_eq!(config.ga.generations, 500);
        assert_eq!(config.simulation.pool_size, 50000);
        // Defaults should still work
        assert_eq!(config.ga.mutation_rate, 0.15);
    }

    #[test]
    fn test_scoring_config_conversion() {
        let config = Config::default();
        let scoring = config.to_scoring_config();
        assert_eq!(scoring.round_scores[0], 1.0);
        assert_eq!(scoring.round_seed_scoring[0], SeedScoring::Add);
        assert_eq!(scoring.round_seed_scoring[5], SeedScoring::Multiply);
    }
}
