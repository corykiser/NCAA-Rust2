//! Tournament field construction.
//!
//! Two sources feed the same structure: the FiveThirtyEight forecast CSV, and a
//! bracket listing whose ratings come from ELO computed on live game results.
//! Both funnel through `TournamentInfo::from_teams`, which validates the field
//! and precomputes everything the optimizers need.

use crate::advancement::AdvancementModel;
use crate::elo::EloSystem;
use crate::game_result::BracketTeam;
use crate::names;
use crate::tree::{CHILDREN, NUM_GAMES, NUM_TEAMS, R1_MATCHUPS, REGION_ORDER};
use csv;
use csv::StringRecord;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Atomically reference-counted Team for efficient sharing without cloning.
/// Arc is used instead of Rc because it's thread-safe for parallel processing with rayon.
pub type RcTeam = Arc<Team>;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Team {
    pub name: String,
    pub seed: i32,
    pub region: String,
    pub rating: f32,
    /// Index into tournament.teams array (0-63) for fast lookup
    #[serde(default)]
    pub team_index: u8,
}

impl Team {
    pub fn new(name: String, seed: i32, region: String, rating: f32) -> Team {
        Team {
            name,
            seed,
            region,
            rating,
            team_index: 0, // Will be set when added to tournament
        }
    }

    pub fn with_index(name: String, seed: i32, region: String, rating: f32, team_index: u8) -> Team {
        Team {
            name,
            seed,
            region,
            rating,
            team_index,
        }
    }
}

impl PartialEq for Team {
    fn eq(&self, other: &Self) -> bool {
        self.team_index == other.team_index
    }
}

/// Pre-computed win probabilities for all team pairs
/// Avoids expensive powf() calls during bracket creation
#[derive(Debug, Clone)]
pub struct ProbabilityCache {
    /// probs[team_a_idx][team_b_idx] = probability that team_a beats team_b
    probs: [[f64; NUM_TEAMS]; NUM_TEAMS],
}

impl ProbabilityCache {
    /// Create cache from team ratings
    pub fn new(teams: &[RcTeam]) -> Self {
        let mut probs = [[0.5f64; NUM_TEAMS]; NUM_TEAMS];

        for (i, team_a) in teams.iter().enumerate() {
            for (j, team_b) in teams.iter().enumerate() {
                if i != j {
                    let rating_diff = team_a.rating as f64 - team_b.rating as f64;
                    probs[i][j] = 1.0 / (1.0 + 10.0f64.powf(-rating_diff * 30.464 / 400.0));
                }
            }
        }

        ProbabilityCache { probs }
    }

    /// Get win probability for team_a vs team_b (using team indices)
    #[inline(always)]
    pub fn get(&self, team_a_idx: u8, team_b_idx: u8) -> f64 {
        self.probs[team_a_idx as usize][team_b_idx as usize]
    }

    /// The same matchups with every probability pushed toward the favourite.
    ///
    /// `tilt > 1` makes the stronger team win more often than the model says.
    /// Used to stand in for a public that picks more chalk than the ratings
    /// justify: running the advancement recurrence over a tilted cache gives
    /// marginals that are automatically consistent, which renormalizing a set
    /// of tilted marginals is not.
    pub fn tilted(&self, tilt: f64) -> ProbabilityCache {
        let mut probs = [[0.5f64; NUM_TEAMS]; NUM_TEAMS];
        for a in 0..NUM_TEAMS {
            for b in 0..NUM_TEAMS {
                if a == b {
                    continue;
                }
                let p = self.probs[a][b];
                let (hi, lo) = (p.powf(tilt), (1.0 - p).powf(tilt));
                probs[a][b] = if hi + lo > 0.0 { hi / (hi + lo) } else { p };
            }
        }
        ProbabilityCache { probs }
    }
}

/// Where a field's ratings come from.
///
/// One field, three interchangeable rating models, so `--ratings` can race them
/// against each other without duplicating the construction path.
pub enum RatingSource<'a> {
    /// Opponent-adjusted least squares over the season game log. The default,
    /// and 0.065 of tournament log loss better than Elo.
    Adjusted(&'a crate::ratings::RidgeRatings),
    /// Elo, kept so the change can be measured rather than asserted.
    Elo(&'a EloSystem),
    /// Seed alone. The backup when no usable game log can be had — degraded
    /// (0.562 against 0.544) but still better than the Elo path it replaces.
    Seed,
}

impl RatingSource<'_> {
    /// The team's rating on the 538 scale, or why it could not be found.
    fn rating_for(&self, team: &BracketTeam) -> Result<f32, String> {
        match self {
            RatingSource::Adjusted(r) => r
                .rating_538_for_name(&team.team_name)
                .map_err(|e| e.to_string()),
            RatingSource::Elo(elo) => elo
                .find_team_by_name(&team.team_name)
                .map(|rated| elo.to_538_scale(&rated.team_id))
                .map_err(|e| e.to_string()),
            RatingSource::Seed => crate::ratings::seed_rating_538(team.seed),
        }
    }

    pub fn describe(&self) -> &'static str {
        match self {
            RatingSource::Adjusted(_) => "opponent-adjusted least squares",
            RatingSource::Elo(_) => "ELO",
            RatingSource::Seed => "seed only (no game data)",
        }
    }
}

/// A rated field, plus the teams that had to fall back to their seed.
#[derive(Debug)]
pub struct RatedField {
    pub info: TournamentInfo,
    /// Formatted lines, one per team rated from its seed. Empty on a clean run;
    /// printed loudly when it is not, because a silent fallback is the failure
    /// mode this codebase most wants to avoid.
    pub fell_back: Vec<String>,
}

#[derive(Debug)]
pub struct TournamentInfo {
    pub teams: Vec<RcTeam>,
    /// Fast lookup map: (region, seed) -> RcTeam
    pub team_lookup: HashMap<(String, i32), RcTeam>,
    /// Pre-computed win probabilities for all team pairs
    pub prob_cache: ProbabilityCache,
    /// Exact per-game advancement probabilities, derived from `prob_cache`.
    pub advancement: AdvancementModel,
    /// Seed of each team, indexed by `team_index`.
    pub seed_of: [i32; NUM_TEAMS],
    /// Alphabetical rank of each team's region (East=0, Midwest=1, South=2, West=3).
    /// The binary encoding breaks cross-region ties by region name, so comparisons
    /// go through this rather than through string comparison in a hot loop.
    pub region_rank: [u8; NUM_TEAMS],
    /// The two teams in each round-1 game, in game order.
    pub r1_teams: [[u8; 2]; 32],
    /// The round-1 game each team plays in, indexed by `team_index`.
    pub r1_game_of_team: [usize; NUM_TEAMS],
    /// `bit_true_beats[w]` has bit `l` set iff a `true` encoding bit in a
    /// matchup between `w` and `l` means `w` advances. Encoding and decoding
    /// run once per game per candidate bracket in the optimizers' inner loops,
    /// so the seed/region comparison is precomputed into a single shift-and-test
    /// against 512 bytes rather than four array loads and a branch chain.
    bit_true_beats: [u64; NUM_TEAMS],
    /// Sampling thresholds over `u32`: `win_threshold[a][b]` is
    /// `P(a beats b) * 2^32`, so drawing an outcome is one integer compare
    /// against a `u32` from the RNG instead of a float conversion and compare.
    pub win_threshold: [[u32; NUM_TEAMS]; NUM_TEAMS],
}

impl TournamentInfo {
    /// Build a tournament from a complete 64-team field.
    ///
    /// The field is validated here rather than trusted: a duplicated
    /// `(region, seed)` pair silently overwrote an entry in the lookup map,
    /// and a duplicated team name made two distinct teams compare equal.
    pub fn from_teams(teams: Vec<RcTeam>) -> Result<TournamentInfo, String> {
        let default_layout = REGION_ORDER.map(|r| r.to_string());
        TournamentInfo::from_teams_with_layout(teams, &default_layout)
    }

    /// Build the field with an explicit Final Four pairing.
    ///
    /// `region_layout[0]` meets `[1]` in one semifinal and `[2]` meets `[3]` in
    /// the other. The NCAA rotates this every year, and it decides which teams
    /// can ever play each other — so it comes from the published bracket when
    /// there is one, rather than from the order the regions happen to be
    /// listed in.
    pub fn from_teams_with_layout(
        mut teams: Vec<RcTeam>,
        region_layout: &[String; 4],
    ) -> Result<TournamentInfo, String> {
        let mut named = region_layout.clone();
        named.sort();
        let mut expected: Vec<String> = REGION_ORDER.iter().map(|r| r.to_string()).collect();
        expected.sort();
        if named.to_vec() != expected {
            return Err(format!(
                "region layout {:?} is not a permutation of {:?}",
                region_layout, REGION_ORDER
            ));
        }

        if teams.len() != NUM_TEAMS {
            return Err(format!(
                "expected {} teams in the field, got {}",
                NUM_TEAMS,
                teams.len()
            ));
        }

        // `team_index` must agree with position, since it indexes every derived table.
        for (i, team) in teams.iter_mut().enumerate() {
            if team.team_index as usize != i {
                let mut fixed = (**team).clone();
                fixed.team_index = i as u8;
                *team = Arc::new(fixed);
            }
        }

        validate_field(&teams)?;

        let mut team_lookup = HashMap::with_capacity(NUM_TEAMS);
        let mut seed_of = [0i32; NUM_TEAMS];
        let mut region_rank = [0u8; NUM_TEAMS];

        let mut region_names: Vec<&str> = REGION_ORDER.to_vec();
        region_names.sort_unstable();

        for team in &teams {
            team_lookup.insert((team.region.clone(), team.seed), Arc::clone(team));
            seed_of[team.team_index as usize] = team.seed;
            region_rank[team.team_index as usize] = region_names
                .iter()
                .position(|r| *r == team.region)
                .expect("region validated above") as u8;
        }

        let mut r1_teams = [[0u8; 2]; 32];
        let mut r1_game_of_team = [usize::MAX; NUM_TEAMS];

        for (region_position, region) in region_layout.iter().enumerate() {
            for (slot, matchup) in R1_MATCHUPS.iter().enumerate() {
                let game = region_position * 8 + slot;
                for (side, &seed) in matchup.iter().enumerate() {
                    let team = team_lookup
                        .get(&(region.clone(), seed))
                        .ok_or_else(|| format!("no team for region {} seed {}", region, seed))?;
                    r1_teams[game][side] = team.team_index;
                    r1_game_of_team[team.team_index as usize] = game;
                }
            }
        }

        let prob_cache = ProbabilityCache::new(&teams);
        let advancement = AdvancementModel::new(&r1_teams, &prob_cache);

        let mut bit_true_beats = [0u64; NUM_TEAMS];
        for w in 0..NUM_TEAMS {
            for l in 0..NUM_TEAMS {
                let same_region = region_rank[w] == region_rank[l];
                let w_is_true = if same_region {
                    seed_of[w] < seed_of[l]
                } else {
                    region_rank[w] < region_rank[l]
                };
                if w_is_true {
                    bit_true_beats[w] |= 1u64 << l;
                }
            }
        }

        let mut win_threshold = [[0u32; NUM_TEAMS]; NUM_TEAMS];
        for a in 0..NUM_TEAMS {
            for b in 0..NUM_TEAMS {
                let p = prob_cache.get(a as u8, b as u8).clamp(0.0, 1.0);
                // `p * 2^32` saturated to `u32::MAX`, so p == 1.0 always wins.
                win_threshold[a][b] = (p * 4_294_967_296.0).min(u32::MAX as f64) as u32;
            }
        }

        Ok(TournamentInfo {
            teams,
            team_lookup,
            prob_cache,
            advancement,
            seed_of,
            region_rank,
            r1_teams,
            r1_game_of_team,
            bit_true_beats,
            win_threshold,
        })
    }

    /// Get a team by region and seed using O(1) lookup
    /// Returns an Arc clone (cheap atomic reference count increment)
    #[inline]
    pub fn get_team(&self, region: &str, seed: i32) -> RcTeam {
        Arc::clone(
            self.team_lookup
                .get(&(region.to_string(), seed))
                .unwrap_or_else(|| panic!("Team not found: region={}, seed={}", region, seed)),
        )
    }

    /// Resolve a team name to its index, refusing to guess between candidates.
    pub fn find_team(&self, name: &str) -> Result<RcTeam, names::NameError> {
        let candidates: Vec<String> = self.teams.iter().map(|t| t.name.clone()).collect();
        names::resolve(name, &candidates).map(|i| Arc::clone(&self.teams[i]))
    }

    /// Which of two teams advances when a game's bit is `true`.
    ///
    /// Within a region that is the numerically lower seed; across regions (the
    /// Final Four and the final) it is the alphabetically earlier region. This
    /// is the single definition of the encoding — everything that reads or
    /// writes a bracket bit goes through here or through `winner_bit`.
    #[inline(always)]
    pub fn bit_true_winner(&self, a: u8, b: u8) -> u8 {
        if self.winner_bit(a, b) {
            a
        } else {
            b
        }
    }

    /// The bit value that makes `winner` beat `loser`.
    #[inline(always)]
    pub fn winner_bit(&self, winner: u8, loser: u8) -> bool {
        self.bit_true_beats[winner as usize] >> loser & 1 != 0
    }

    /// Decode a 63-bit bracket into the winning team index of each game.
    ///
    /// Cheaper than building a full `Bracket` when only the winners are needed —
    /// no `Game` structs and no reference counting.
    pub fn decode_winners(&self, binary: &[bool]) -> [u8; NUM_GAMES] {
        debug_assert_eq!(binary.len(), NUM_GAMES);
        let mut winners = [0u8; NUM_GAMES];

        for game in 0..NUM_GAMES {
            let (a, b) = self.participants(game, &winners);
            let bit_true = self.bit_true_winner(a, b);
            winners[game] = if binary[game] {
                bit_true
            } else if bit_true == a {
                b
            } else {
                a
            };
        }

        winners
    }

    /// The two teams playing in `game`, given the winners of everything below it.
    #[inline]
    pub fn participants(&self, game: usize, winners: &[u8; NUM_GAMES]) -> (u8, u8) {
        if game < 32 {
            (self.r1_teams[game][0], self.r1_teams[game][1])
        } else {
            let [c0, c1] = CHILDREN[game];
            (winners[c0], winners[c1])
        }
    }

    /// Encode a full set of game winners back into the 63-bit representation.
    pub fn binary_from_winners(&self, winners: &[u8; NUM_GAMES]) -> Vec<bool> {
        (0..NUM_GAMES)
            .map(|game| {
                let (a, b) = self.participants(game, winners);
                let winner = winners[game];
                debug_assert!(
                    winner == a || winner == b,
                    "game {} winner {} is not a participant ({}, {})",
                    game,
                    winner,
                    a,
                    b
                );
                let loser = if winner == a { b } else { a };
                self.winner_bit(winner, loser)
            })
            .collect()
    }

    /// Load the tournament field from the FiveThirtyEight forecast CSV.
    pub fn initialize(file_path: &str) -> Result<TournamentInfo, String> {
        let mut rdr =
            csv::Reader::from_path(file_path).map_err(|e| format!("{}: {}", file_path, e))?;

        let mut records: Vec<StringRecord> = Vec::new();
        for result in rdr.records() {
            let record = result.map_err(|e| format!("{}: {}", file_path, e))?;
            if record[0].starts_with("mens")
                && record[1].contains("2023-03-15")
                && record[3].contains("1.0")
            {
                records.push(record);
            }
        }

        let mut teams: Vec<RcTeam> = Vec::with_capacity(NUM_TEAMS);
        for (i, record) in records.iter().enumerate() {
            let rating: f64 = record[14]
                .parse()
                .map_err(|_| format!("row {}: bad rating '{}'", i, &record[14]))?;
            let name = record[13].to_string();
            let region = record[15].to_string();

            // Play-in seeds are written "11a"/"11b"; both halves are the same seed.
            let seed_text = record[16].trim_end_matches(|c: char| c.is_ascii_alphabetic());
            let seed: i32 = seed_text
                .parse()
                .map_err(|_| format!("row {}: bad seed '{}'", i, &record[16]))?;

            teams.push(Arc::new(Team::with_index(
                name,
                seed,
                region,
                rating as f32,
                i as u8,
            )));
        }

        TournamentInfo::from_teams(teams)
    }

    /// Build the tournament from a bracket listing, rating each team with ELO
    /// computed from game results.
    ///
    /// Every name must resolve to exactly one rated team. A team that cannot be
    /// resolved is an error rather than a default rating: a silent 75.0 turns a
    /// contender into a coin flip and nothing downstream can tell.
    /// Attach ratings to a field, using the bracket's real Final Four pairing.
    ///
    /// The layout is not optional: defaulting it to the order the regions are
    /// listed in is the assumption that got the 2026 semifinals wrong, and it
    /// fails silently.
    pub fn from_elo_ratings_with_layout(
        elo_system: &EloSystem,
        bracket_teams: Vec<BracketTeam>,
        region_layout: &[String; 4],
    ) -> Result<TournamentInfo, String> {
        TournamentInfo::from_rating_source_with_layout(
            &RatingSource::Elo(elo_system),
            bracket_teams,
            region_layout,
            0,
        )
        .map(|rated| rated.info)
    }

    /// Attach ratings from any source to a field.
    ///
    /// `max_seed_fallbacks` is how many individual teams may be rated from their
    /// seed when their name will not resolve against the rating source. A
    /// handful is a spelling the alias table has not seen; a field full of them
    /// means the two feeds disagree systematically, and rating the whole
    /// tournament off seed numbers while claiming to have used a season of games
    /// is exactly the kind of quiet wrongness this codebase is built to refuse.
    pub fn from_rating_source_with_layout(
        source: &RatingSource,
        bracket_teams: Vec<BracketTeam>,
        region_layout: &[String; 4],
        max_seed_fallbacks: usize,
    ) -> Result<RatedField, String> {
        let mut teams: Vec<RcTeam> = Vec::with_capacity(bracket_teams.len());
        let mut unresolved: Vec<String> = Vec::new();
        let mut fell_back: Vec<String> = Vec::new();

        for (idx, bracket_team) in bracket_teams.iter().enumerate() {
            let rating = match source.rating_for(bracket_team) {
                Ok(rating) => rating,
                Err(why) => {
                    // Seeds are always available — they come from the same
                    // bracket listing as the name — so the per-team fallback
                    // never fails for a reason the field itself can fix.
                    match crate::ratings::seed_rating_538(bracket_team.seed) {
                        Ok(rating) if fell_back.len() < max_seed_fallbacks => {
                            fell_back.push(format!(
                                "  {} ({} seed): {}",
                                bracket_team.team_name, bracket_team.seed, why
                            ));
                            rating
                        }
                        _ => {
                            unresolved.push(format!("  {}: {}", bracket_team.team_name, why));
                            0.0
                        }
                    }
                }
            };

            teams.push(Arc::new(Team::with_index(
                bracket_team.team_name.clone(),
                bracket_team.seed,
                bracket_team.region.clone(),
                rating,
                idx as u8,
            )));
        }

        if !unresolved.is_empty() {
            return Err(format!(
                "{} bracket team(s) could not be matched to a rating:\n{}\n\n\
                 Up to {} of these can be rated from their seed instead; past \
                 that the two feeds disagree systematically and the ratings \
                 would be wrong in a way nothing downstream could detect.",
                unresolved.len(),
                unresolved.join("\n"),
                max_seed_fallbacks,
            ));
        }

        Ok(RatedField {
            info: TournamentInfo::from_teams_with_layout(teams, region_layout)?,
            fell_back,
        })
    }

    /// A sample field, used when no bracket source is available.
    /// Mirrors the 2024 tournament.
    pub fn sample_bracket_teams() -> Vec<BracketTeam> {
        let sample_teams_by_region = [
            (
                "East",
                [
                    ("Connecticut", 1),
                    ("Iowa State", 2),
                    ("Illinois", 3),
                    ("Auburn", 4),
                    ("San Diego State", 5),
                    ("BYU", 6),
                    ("Washington State", 7),
                    ("Florida Atlantic", 8),
                    ("Northwestern", 9),
                    ("Drake", 10),
                    ("Duquesne", 11),
                    ("UAB", 12),
                    ("Yale", 13),
                    ("Morehead State", 14),
                    ("Long Beach State", 15),
                    ("Stetson", 16),
                ],
            ),
            (
                "West",
                [
                    ("North Carolina", 1),
                    ("Arizona", 2),
                    ("Baylor", 3),
                    ("Alabama", 4),
                    ("Saint Mary's", 5),
                    ("Clemson", 6),
                    ("Dayton", 7),
                    ("Mississippi State", 8),
                    ("Michigan State", 9),
                    ("Nevada", 10),
                    ("New Mexico", 11),
                    ("Grand Canyon", 12),
                    ("Charleston", 13),
                    ("Colgate", 14),
                    ("Long Island", 15),
                    ("Wagner", 16),
                ],
            ),
            (
                "South",
                [
                    ("Houston", 1),
                    ("Marquette", 2),
                    ("Kentucky", 3),
                    ("Duke", 4),
                    ("Wisconsin", 5),
                    ("Texas Tech", 6),
                    ("Florida", 7),
                    ("Nebraska", 8),
                    ("Texas A&M", 9),
                    ("Colorado", 10),
                    ("NC State", 11),
                    ("James Madison", 12),
                    ("Vermont", 13),
                    ("Oakland", 14),
                    ("Western Kentucky", 15),
                    ("Longwood", 16),
                ],
            ),
            (
                "Midwest",
                [
                    ("Purdue", 1),
                    ("Tennessee", 2),
                    ("Creighton", 3),
                    ("Kansas", 4),
                    ("Gonzaga", 5),
                    ("South Carolina", 6),
                    ("Texas", 7),
                    ("Utah State", 8),
                    ("TCU", 9),
                    ("Colorado State", 10),
                    ("Oregon", 11),
                    ("McNeese", 12),
                    ("Samford", 13),
                    ("Akron", 14),
                    ("Grambling State", 15),
                    ("Montana State", 16),
                ],
            ),
        ];

        let mut teams = Vec::with_capacity(NUM_TEAMS);
        for (region, region_teams) in &sample_teams_by_region {
            for (name, seed) in region_teams {
                teams.push(BracketTeam::new(
                    name.to_lowercase().replace(' ', "-"),
                    name.to_string(),
                    *seed,
                    region.to_string(),
                ));
            }
        }
        teams
    }
}

/// Reject a field that would silently misbehave downstream.
fn validate_field(teams: &[RcTeam]) -> Result<(), String> {
    let mut problems: Vec<String> = Vec::new();

    let mut seen_slots: HashMap<(&str, i32), &str> = HashMap::new();
    let mut seen_names: HashMap<String, &str> = HashMap::new();
    let mut region_counts: HashMap<&str, usize> = HashMap::new();

    for team in teams {
        if !REGION_ORDER.contains(&team.region.as_str()) {
            problems.push(format!(
                "{}: unknown region '{}' (expected one of {:?})",
                team.name, team.region, REGION_ORDER
            ));
            continue;
        }
        *region_counts.entry(team.region.as_str()).or_insert(0) += 1;

        if !(1..=16).contains(&team.seed) {
            problems.push(format!("{}: seed {} is outside 1-16", team.name, team.seed));
        }

        if let Some(other) = seen_slots.insert((team.region.as_str(), team.seed), &team.name) {
            problems.push(format!(
                "{} and {} are both the {} seed in the {}",
                other, team.name, team.seed, team.region
            ));
        }

        let normalized = names::normalize(&team.name);
        if let Some(other) = seen_names.insert(normalized, &team.name) {
            problems.push(format!(
                "duplicate team name: '{}' and '{}'",
                other, team.name
            ));
        }

        if !team.rating.is_finite() {
            problems.push(format!("{}: rating is not a finite number", team.name));
        }
    }

    for region in REGION_ORDER {
        let count = region_counts.get(region).copied().unwrap_or(0);
        if count != 16 {
            problems.push(format!("{} region has {} teams, expected 16", region, count));
        }
    }

    let indices: HashSet<u8> = teams.iter().map(|t| t.team_index).collect();
    if indices.len() != teams.len() {
        problems.push("team indices are not unique".to_string());
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "invalid tournament field:\n  {}",
            problems.join("\n  ")
        ))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn field() -> Vec<RcTeam> {
        let mut teams = Vec::new();
        for (r, region) in REGION_ORDER.iter().enumerate() {
            for seed in 1..=16 {
                let idx = (r * 16 + seed as usize - 1) as u8;
                teams.push(Arc::new(Team::with_index(
                    format!("{} {}", region, seed),
                    seed,
                    region.to_string(),
                    // Strictly decreasing in seed so the favourite is unambiguous.
                    100.0 - seed as f32,
                    idx,
                )));
            }
        }
        teams
    }

    fn bracket_field() -> Vec<BracketTeam> {
        let mut teams = Vec::new();
        for region in REGION_ORDER.iter() {
            for seed in 1..=16 {
                // Zero-padded on purpose: "East 1" is a prefix of "East 10",
                // and `names::resolve` matches on prefixes, so unpadded names
                // would make this test measure the resolver rather than the
                // fallback cap.
                teams.push(BracketTeam::new(
                    format!("{}-{}", region, seed),
                    format!("{} {:02}", region, seed),
                    seed,
                    region.to_string(),
                ));
            }
        }
        teams
    }

    /// The whole point of the seed source: a field with no game data behind it
    /// still produces a legal tournament with the favourite favoured.
    #[test]
    fn the_seed_source_rates_a_field_with_no_game_data() {
        let layout = REGION_ORDER.map(|r| r.to_string());
        let rated = TournamentInfo::from_rating_source_with_layout(
            &RatingSource::Seed,
            bracket_field(),
            &layout,
            0,
        )
        .expect("seeds are always available");
        assert!(rated.fell_back.is_empty(), "seeds never fall back");

        let one = rated.info.team_lookup[&("East".to_string(), 1)].clone();
        let sixteen = rated.info.team_lookup[&("East".to_string(), 16)].clone();
        let p = rated
            .info
            .prob_cache
            .get(one.team_index, sixteen.team_index);
        assert!(p > 0.9 && p < 0.995, "1 over 16: {}", p);
    }

    /// A name the rating source has never heard of falls back to its seed, up to
    /// the cap — and past the cap the whole field is refused rather than quietly
    /// rated off seed numbers while claiming a season of games.
    #[test]
    fn unresolvable_names_fall_back_to_seed_only_up_to_the_cap() {
        let layout = REGION_ORDER.map(|r| r.to_string());
        let empty = crate::ratings::fit(&[], crate::ratings::RIDGE_LAMBDA);
        assert!(empty.is_err(), "an empty log cannot be fitted");

        // A fit that knows exactly one team: every other name is unresolvable.
        let games = vec![crate::game_result::GameResult {
            game_id: "g".into(),
            date: chrono::NaiveDate::from_ymd_opt(2025, 11, 3).unwrap(),
            home_team_id: "east01".into(),
            home_team_name: "East 01".into(),
            away_team_id: "west01".into(),
            away_team_name: "West 01".into(),
            home_score: 80,
            away_score: 70,
            is_neutral_site: true,
            is_conference_game: false,
            is_completed: true,
        }];
        let fit = crate::ratings::fit(&games, crate::ratings::RIDGE_LAMBDA).expect("fits");

        let capped = TournamentInfo::from_rating_source_with_layout(
            &RatingSource::Adjusted(&fit),
            bracket_field(),
            &layout,
            4,
        );
        let err = capped.expect_err("62 unresolvable names is a systematic failure");
        assert!(err.contains("could not be matched"), "{}", err);

        // With room for all of them, every unresolved team is reported by name.
        let permissive = TournamentInfo::from_rating_source_with_layout(
            &RatingSource::Adjusted(&fit),
            bracket_field(),
            &layout,
            64,
        )
        .expect("everything falls back");
        // Not an exact count: `names::resolve` also matches on substrings, so
        // "Midwest 01" resolves to the rated "West 01" too. What matters is that
        // the overwhelming majority fell back and that a team the fit actually
        // knows kept its rating.
        assert!(
            permissive.fell_back.len() >= 60,
            "nearly the whole field should fall back, got {}",
            permissive.fell_back.len()
        );
        assert!(
            !permissive
                .fell_back
                .iter()
                // The prefix, not a `contains`: every other line names "East 01"
                // as a suggestion.
                .any(|line| line.trim_start().starts_with("East 01 ")),
            "a team the fit knows keeps its rating: {:?}",
            permissive.fell_back
        );
    }

    /// The Final Four pairing decides which regions can meet, so the layout
    /// has to reach the round-1 placement tables.
    #[test]
    fn the_region_layout_places_regions_in_bracket_order() {
        let layout = [
            "East".to_string(),
            "South".to_string(),
            "West".to_string(),
            "Midwest".to_string(),
        ];
        let info = TournamentInfo::from_teams_with_layout(field(), &layout).expect("valid field");

        // Games 0-7 are the first region in the layout, 8-15 the second, and
        // the two meet in a semifinal.
        for (block, region) in layout.iter().enumerate() {
            for game in block * 8..block * 8 + 8 {
                for side in 0..2 {
                    let team = &info.teams[info.r1_teams[game][side] as usize];
                    assert_eq!(&team.region, region, "game {} side {}", game, side);
                }
            }
        }
    }

    #[test]
    fn a_layout_that_is_not_the_four_regions_is_rejected() {
        let layout = [
            "East".to_string(),
            "East".to_string(),
            "West".to_string(),
            "Midwest".to_string(),
        ];
        let err = TournamentInfo::from_teams_with_layout(field(), &layout).unwrap_err();
        assert!(err.contains("permutation"), "{}", err);
    }

    pub fn tournament() -> TournamentInfo {
        TournamentInfo::from_teams(field()).expect("valid field")
    }

    #[test]
    fn duplicate_seeds_in_a_region_are_rejected() {
        let mut teams = field();
        let clash = Team::with_index("East 1 again".into(), 1, "East".into(), 90.0, 5);
        teams[5] = Arc::new(clash);
        let err = TournamentInfo::from_teams(teams).unwrap_err();
        assert!(err.contains("1 seed in the East"), "{}", err);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut teams = field();
        let dup = Team::with_index("East 1".into(), 7, "West".into(), 90.0, 22);
        teams[22] = Arc::new(dup);
        let err = TournamentInfo::from_teams(teams).unwrap_err();
        assert!(err.contains("duplicate team name"), "{}", err);
    }

    #[test]
    fn the_shipped_sample_field_is_valid() {
        // It previously listed "Texas" in two different regions.
        let sample = TournamentInfo::sample_bracket_teams();
        let teams: Vec<RcTeam> = sample
            .iter()
            .enumerate()
            .map(|(i, t)| {
                Arc::new(Team::with_index(
                    t.team_name.clone(),
                    t.seed,
                    t.region.clone(),
                    80.0 - t.seed as f32,
                    i as u8,
                ))
            })
            .collect();
        TournamentInfo::from_teams(teams).expect("sample field should be valid");
    }

    #[test]
    fn every_team_appears_in_exactly_one_round1_game() {
        let t = tournament();
        let mut seen = [0usize; NUM_TEAMS];
        for game in &t.r1_teams {
            for &team in game {
                seen[team as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        assert!(t.r1_game_of_team.iter().all(|&g| g < 32));
    }

    #[test]
    fn round1_games_pair_seeds_that_sum_to_seventeen() {
        let t = tournament();
        for game in &t.r1_teams {
            let (a, b) = (game[0], game[1]);
            assert_eq!(t.seed_of[a as usize] + t.seed_of[b as usize], 17);
            assert_eq!(t.region_rank[a as usize], t.region_rank[b as usize]);
        }
    }

    #[test]
    fn decoding_and_encoding_a_bracket_round_trips() {
        let t = tournament();
        let mut rng_state = 0x243f6a8885a308d3u64;
        for _ in 0..200 {
            let binary: Vec<bool> = (0..NUM_GAMES)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    rng_state >> 63 == 1
                })
                .collect();
            let winners = t.decode_winners(&binary);
            assert_eq!(t.binary_from_winners(&winners), binary);
        }
    }

    #[test]
    fn decoded_winners_are_always_participants() {
        let t = tournament();
        let binary = vec![true; NUM_GAMES];
        let winners = t.decode_winners(&binary);
        for game in 0..NUM_GAMES {
            let (a, b) = t.participants(game, &winners);
            assert!(winners[game] == a || winners[game] == b);
        }
        // All-true means the favourite by seed wins every intra-region game, so
        // the four 1-seeds reach the Elite 8.
        for region in 0..4 {
            assert_eq!(t.seed_of[winners[56 + region] as usize], 1);
        }
    }
}
