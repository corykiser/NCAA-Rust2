//! The rest of the pool: what everybody *else* picked.
//!
//! Every objective before this one scored a bracket against the tournament
//! alone. That is the right thing to maximize only if your payout is linear in
//! points. In a pool that pays the top finisher, what you want is
//! `P(one of my entries finishes first)`, and that depends on the competition:
//! a bracket is worth less when thousands of other entrants submitted it too.
//!
//! The public's picks are published as *marginals* — the fraction of entries
//! that picked each team to reach each round. ESPN's Tournament Challenge
//! exposes them for all six rounds and all 64 teams; in 2023, 18.9% of roughly
//! 20 million entries picked Alabama to win it all. That is the number that
//! makes chalk expensive: being right about the favourite is worth little when
//! you split first place with a fifth of the field.
//!
//! Marginals are not a joint distribution, so this reconstructs one. Walking
//! the bracket, at each game between the two teams the walk has delivered, an
//! entry advances team `x` with probability proportional to `x`'s *conditional*
//! advance rate — the share of entries that had `x` winning this round given
//! they had it winning the last. That reproduces the published marginals
//! closely while producing whole, legal brackets to compete against.

use crate::advancement::AdvancementModel;
use crate::ingest::TournamentInfo;
use crate::names;
use crate::picks::Picks;
use crate::tree::{NUM_ROUNDS, NUM_TEAMS, ROUND_GAMES, ROUND_OF, ROUND_START};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;

/// Smallest conditional advance rate considered. A team no entry advanced still
/// has to be pickable, or a scenario where it survives has no competition in it
/// at all and looks like a free win.
const FLOOR: f32 = 1e-4;

/// One team's line in a pick-popularity file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamPicks {
    pub name: String,
    pub seed: i32,
    /// The source's own region identifier. Only its grouping matters, not its
    /// value — see `PickPopularity::from_file`.
    #[serde(default)]
    pub region_id: i32,
    /// `reach[r]` = fraction of public entries picking this team to win at
    /// least `r + 1` games. Six entries, round of 32 through champion.
    pub reach: Vec<f64>,
}

/// A pick-popularity file, as fetched or hand-written.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PickPopularityFile {
    pub source: String,
    #[serde(default)]
    pub sample_size: Option<u64>,
    pub teams: Vec<TeamPicks>,
}

/// Public pick rates, resolved onto this tournament's team indices.
#[derive(Debug, Clone)]
pub struct PickPopularity {
    /// `reach[round][team]` = fraction of entries picking `team` to win at
    /// least `round + 1` games.
    reach: [[f32; NUM_TEAMS]; NUM_ROUNDS],
    pub source: String,
    pub sample_size: Option<u64>,
}

impl PickPopularity {
    /// Resolve a popularity file against a tournament field.
    ///
    /// Names come from a different source than the ratings do, so they go
    /// through the same resolver the `--lock-team` flag uses, and an
    /// unmatched name is an error rather than a silently dropped team.
    pub fn from_file(
        file: &PickPopularityFile,
        tournament: &TournamentInfo,
    ) -> Result<PickPopularity, String> {
        if file.teams.len() != NUM_TEAMS {
            return Err(format!(
                "expected {} teams of pick data, got {}",
                NUM_TEAMS,
                file.teams.len()
            ));
        }
        for entry in &file.teams {
            if entry.reach.len() != NUM_ROUNDS {
                return Err(format!(
                    "{}: expected {} reach values, got {}",
                    entry.name,
                    NUM_ROUNDS,
                    entry.reach.len()
                ));
            }
        }

        let assignment = resolve_teams(&file.teams, tournament)?;

        let mut reach = [[0.0f32; NUM_TEAMS]; NUM_ROUNDS];
        for (entry, &index) in file.teams.iter().zip(assignment.iter()) {
            for round in 0..NUM_ROUNDS {
                reach[round][index] = entry.reach[round].clamp(0.0, 1.0) as f32;
            }
        }

        let mut popularity = PickPopularity {
            reach,
            source: file.source.clone(),
            sample_size: file.sample_size,
        };
        popularity.enforce_nesting();
        Ok(popularity)
    }

    pub fn load(path: &str, tournament: &TournamentInfo) -> Result<PickPopularity, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        let file: PickPopularityFile =
            serde_json::from_str(&text).map_err(|e| format!("{}: {}", path, e))?;
        Self::from_file(&file, tournament)
    }

    /// Stand-in for when no published picks are available.
    ///
    /// Public entries are more favourite-heavy than the rating model is, so
    /// this pushes every *game* probability toward the favourite and runs the
    /// ordinary advancement recurrence over the result. Tilting the marginals
    /// directly instead does not work: a favourite's round-1 share clamps at
    /// 1.0, the clamped mass vanishes, and the round stops summing to its
    /// number of survivors.
    ///
    /// It is a shape, not data — worth replacing with the real thing, because
    /// how *concentrated* the public is on the favourite is exactly what
    /// decides whether picking the favourite is a good idea.
    pub fn chalk(tournament: &TournamentInfo, tilt: f64) -> PickPopularity {
        let public_view = tournament.prob_cache.tilted(tilt);
        let advancement = AdvancementModel::new(&tournament.r1_teams, &public_view);

        let mut reach = [[0.0f32; NUM_TEAMS]; NUM_ROUNDS];
        for round in 0..NUM_ROUNDS {
            for game in ROUND_START[round]..ROUND_START[round] + ROUND_GAMES[round] {
                for &team in advancement.subtree(game) {
                    // A team plays at most one game per round, so this assigns
                    // rather than accumulates across games.
                    reach[round][team as usize] += advancement.win_prob(game, team) as f32;
                }
            }
        }

        let mut popularity = PickPopularity {
            reach,
            source: format!("chalk model (tilt {:.2})", tilt),
            sample_size: None,
        };
        popularity.enforce_nesting();
        popularity
    }

    /// A team cannot reach the Final Four in more entries than it reaches the
    /// Sweet 16 in. Sampling noise and renormalization can both break that;
    /// the sampler divides these values, so fix it up front.
    fn enforce_nesting(&mut self) {
        for round in 1..NUM_ROUNDS {
            for team in 0..NUM_TEAMS {
                let ceiling = self.reach[round - 1][team];
                if self.reach[round][team] > ceiling {
                    self.reach[round][team] = ceiling;
                }
            }
        }
    }

    /// Fraction of public entries picking `team` to win at least `round + 1` games.
    #[inline]
    #[cfg(test)]
    pub fn reach(&self, round: usize, team: u8) -> f32 {
        self.reach[round][team as usize]
    }

    /// Share of entries that had `team` winning `round`, given they had it
    /// winning the round before.
    #[inline]
    fn advance_rate(&self, round: usize, team: u8) -> f32 {
        let numerator = self.reach[round][team as usize];
        let denominator = if round == 0 {
            1.0
        } else {
            self.reach[round - 1][team as usize]
        };
        if denominator <= 0.0 {
            return FLOOR;
        }
        (numerator / denominator).clamp(FLOOR, 1.0)
    }

    /// Draw one public entry.
    pub fn sample_entry(&self, tournament: &TournamentInfo, rng: &mut impl Rng) -> Picks {
        Picks::sample_with(tournament, rng, |game, a, b| {
            let round = ROUND_OF[game];
            let qa = self.advance_rate(round, a);
            let qb = self.advance_rate(round, b);
            qa / (qa + qb)
        })
    }

    /// The most-picked champion, and the share of entries that picked them.
    pub fn favourite(&self) -> (u8, f32) {
        let mut best = (0u8, -1.0f32);
        for team in 0..NUM_TEAMS {
            let share = self.reach[NUM_ROUNDS - 1][team];
            if share > best.1 {
                best = (team as u8, share);
            }
        }
        best
    }
}

/// Map each row of pick data onto a team index.
///
/// Names are the obvious join and the wrong one: the pick source calls them
/// "FAU", "UConn" and "Texas A&M-CC" where the ratings source says "Florida
/// Atlantic", "Connecticut" and "Texas A&M-Corpus Christi". An alias table
/// would need a new entry every time a source changes its abbreviations.
///
/// `(region, seed)` *is* unique, so the join uses that instead. The only thing
/// missing is which of the source's region identifiers is which region, and
/// that can be learned rather than configured: resolve the names that are
/// unambiguous within their seed — a seed has only four to six teams, so most
/// of them are — and let those matches vote on the region mapping. A handful of
/// confident matches pins all four regions, and every remaining team, however
/// it is abbreviated, then follows exactly.
fn resolve_teams(
    rows: &[TeamPicks],
    tournament: &TournamentInfo,
) -> Result<Vec<usize>, String> {
    // Teams of the tournament, grouped by seed and by (region, seed).
    let mut by_seed: Vec<Vec<usize>> = vec![Vec::new(); 17];
    let mut by_region_seed: std::collections::HashMap<(u8, i32), usize> =
        std::collections::HashMap::with_capacity(NUM_TEAMS);
    for team in &tournament.teams {
        let index = team.team_index as usize;
        by_seed[team.seed as usize].push(index);
        by_region_seed.insert((tournament.region_rank[index], team.seed), index);
    }

    // Pass 1: confident, seed-scoped name matches vote on the region mapping.
    let mut votes: std::collections::HashMap<(i32, u8), usize> =
        std::collections::HashMap::new();
    for row in rows {
        let peers = match by_seed.get(row.seed as usize) {
            Some(p) if !p.is_empty() => p,
            _ => continue,
        };
        let names: Vec<String> = peers
            .iter()
            .map(|&i| tournament.teams[i].name.clone())
            .collect();
        if let Ok(local) = names::resolve(&row.name, &names) {
            let region = tournament.region_rank[peers[local]];
            *votes.entry((row.region_id, region)).or_insert(0) += 1;
        }
    }

    // Each source region maps to whichever tournament region it voted for most.
    let mut source_regions: Vec<i32> = rows.iter().map(|r| r.region_id).collect();
    source_regions.sort_unstable();
    source_regions.dedup();

    let mut mapping: std::collections::HashMap<i32, u8> = std::collections::HashMap::new();
    if source_regions.len() == 4 {
        for &source in &source_regions {
            let winner = (0..4u8)
                .map(|region| (region, votes.get(&(source, region)).copied().unwrap_or(0)))
                .max_by_key(|&(_, count)| count);
            if let Some((region, count)) = winner {
                if count > 0 {
                    mapping.insert(source, region);
                }
            }
        }
        // A mapping that is not a bijection is not a mapping.
        let mut targets: Vec<u8> = mapping.values().copied().collect();
        targets.sort_unstable();
        targets.dedup();
        if mapping.len() != 4 || targets.len() != 4 {
            mapping.clear();
        }
    }

    // Pass 2: assign by (region, seed) where the mapping is known, by name
    // where it is not.
    let mut assignment = vec![usize::MAX; rows.len()];
    let mut taken = [false; NUM_TEAMS];
    let all_names: Vec<String> = tournament.teams.iter().map(|t| t.name.clone()).collect();
    let mut unresolved: Vec<&str> = Vec::new();

    for (row_index, row) in rows.iter().enumerate() {
        let index = match mapping.get(&row.region_id) {
            Some(&region) => by_region_seed.get(&(region, row.seed)).copied(),
            None => {
                let peers = by_seed.get(row.seed as usize).cloned().unwrap_or_default();
                let names: Vec<String> = peers
                    .iter()
                    .map(|&i| tournament.teams[i].name.clone())
                    .collect();
                names::resolve(&row.name, &names)
                    .map(|local| peers[local])
                    .or_else(|_| names::resolve(&row.name, &all_names))
                    .ok()
            }
        };

        match index {
            Some(i) if !taken[i] => {
                taken[i] = true;
                assignment[row_index] = i;
            }
            _ => unresolved.push(&row.name),
        }
    }

    if !unresolved.is_empty() {
        return Err(format!(
            "could not place {} team(s) from the pick data onto the tournament \
             field: {}. Check that the pick data is for the same tournament.",
            unresolved.len(),
            unresolved.join(", ")
        ));
    }

    Ok(assignment)
}

/// Fetch published pick rates from ESPN's Tournament Challenge.
///
/// Each scoring period's payload lists every one of the 64 teams with the share
/// of entries that picked it to survive that round, which is exactly the
/// marginal this module wants — no scraping of the rendered page required.
pub fn fetch_espn(year: i32, cache_dir: &str) -> Result<PickPopularityFile, String> {
    let cache_path = format!("{}/picks_{}.json", cache_dir, year);
    if let Ok(text) = fs::read_to_string(&cache_path) {
        if let Ok(file) = serde_json::from_str::<PickPopularityFile>(&text) {
            return Ok(file);
        }
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    // name -> (seed, region id, one reach value per round)
    let mut rows: std::collections::HashMap<String, (i32, i32, [f64; NUM_ROUNDS])> =
        std::collections::HashMap::new();
    let mut sample_size: Option<u64> = None;

    for round in 0..NUM_ROUNDS {
        let url = format!(
            "https://gambit-api.fantasy.espn.com/apis/v1/challenges/\
             tournament-challenge-bracket-{}?scoringPeriodId={}&view=chui_default",
            year,
            round + 1
        );
        let body: serde_json::Value = client
            .get(&url)
            .send()
            .map_err(|e| format!("ESPN picks for {}: {}", year, e))?
            .json()
            .map_err(|e| format!("ESPN picks for {}: {}", year, e))?;

        let propositions = body
            .get("propositions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("ESPN picks for {}: no propositions", year))?;

        for proposition in propositions {
            let outcomes = match proposition.get("possibleOutcomes").and_then(|v| v.as_array()) {
                Some(o) => o,
                None => continue,
            };
            for outcome in outcomes {
                let name = match outcome.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => continue,
                };
                let seed = outcome
                    .get("regionSeed")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                let region_id = outcome
                    .get("regionId")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                for counter in outcome
                    .get("choiceCounters")
                    .and_then(|v| v.as_array())
                    .unwrap_or(&Vec::new())
                {
                    if let Some(pct) = counter.get("percentage").and_then(|v| v.as_f64()) {
                        rows.entry(name.to_string())
                            .or_insert((seed, region_id, [0.0; NUM_ROUNDS]))
                            .2[round] = pct;
                    }
                    if round == 0 {
                        if let Some(count) = counter.get("count").and_then(|v| v.as_u64()) {
                            // Round-1 counts are per matchup side; the largest
                            // is a floor on how many entries there were.
                            sample_size = Some(sample_size.unwrap_or(0).max(count));
                        }
                    }
                }
            }
        }
    }

    if rows.len() != NUM_TEAMS {
        return Err(format!(
            "ESPN picks for {}: got {} teams, expected {}",
            year,
            rows.len(),
            NUM_TEAMS
        ));
    }

    let mut teams: Vec<TeamPicks> = rows
        .into_iter()
        .map(|(name, (seed, region_id, reach))| TeamPicks {
            name,
            seed,
            region_id,
            reach: reach.to_vec(),
        })
        .collect();
    teams.sort_by(|a, b| a.name.cmp(&b.name));

    let file = PickPopularityFile {
        source: format!("espn:{}", year),
        sample_size,
        teams,
    };

    if let Err(e) = fs::create_dir_all(cache_dir) {
        eprintln!("could not create {}: {}", cache_dir, e);
    } else if let Ok(json) = serde_json::to_string_pretty(&file) {
        if let Err(e) = fs::write(&cache_path, json) {
            eprintln!("could not cache {}: {}", cache_path, e);
        }
    }

    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::tests::tournament;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn chalk(t: &TournamentInfo) -> PickPopularity {
        PickPopularity::chalk(t, 1.5)
    }

    #[test]
    fn each_round_has_the_right_number_of_survivors() {
        let t = tournament();
        let p = chalk(&t);
        for round in 0..NUM_ROUNDS {
            let total: f32 = (0..NUM_TEAMS).map(|i| p.reach(round, i as u8)).sum();
            assert!(
                (total - ROUND_GAMES[round] as f32).abs() < 0.05,
                "round {} marginals sum to {}, expected {}",
                round,
                total,
                ROUND_GAMES[round]
            );
        }
    }

    #[test]
    fn reach_never_increases_with_round() {
        let t = tournament();
        let p = chalk(&t);
        for team in 0..NUM_TEAMS as u8 {
            for round in 1..NUM_ROUNDS {
                assert!(
                    p.reach(round, team) <= p.reach(round - 1, team) + 1e-6,
                    "team {} reaches round {} more often than round {}",
                    team,
                    round,
                    round - 1
                );
            }
        }
    }

    #[test]
    fn sampled_entries_are_legal_brackets() {
        let t = tournament();
        let p = chalk(&t);
        let mut rng = SmallRng::seed_from_u64(4);
        for _ in 0..500 {
            let entry = p.sample_entry(&t, &mut rng);
            // Round-trips only if the winners and bits agree, which in turn
            // only holds for a structurally legal bracket.
            assert_eq!(Picks::from_bits(&t, entry.bits()), entry);
        }
    }

    #[test]
    fn sampled_entries_reproduce_the_published_marginals() {
        // The sampler turns marginals into whole brackets; the brackets it
        // makes have to carry the marginals back out, or the field it builds
        // is not the field that was measured.
        let t = tournament();
        let p = chalk(&t);
        let mut rng = SmallRng::seed_from_u64(2024);

        const DRAWS: usize = 20_000;
        let mut seen = [[0u32; NUM_TEAMS]; NUM_ROUNDS];
        for _ in 0..DRAWS {
            let entry = p.sample_entry(&t, &mut rng);
            for round in 0..NUM_ROUNDS {
                for game in ROUND_START[round]..ROUND_START[round] + ROUND_GAMES[round] {
                    seen[round][entry.winner(game) as usize] += 1;
                }
            }
        }

        for round in 0..NUM_ROUNDS {
            for team in 0..NUM_TEAMS {
                let observed = seen[round][team] as f32 / DRAWS as f32;
                let expected = p.reach(round, team as u8);
                assert!(
                    (observed - expected).abs() < 0.05,
                    "round {} team {}: sampled {:.3}, published {:.3}",
                    round,
                    team,
                    observed,
                    expected
                );
            }
        }
    }

    #[test]
    fn a_tilt_above_one_concentrates_the_public_on_favourites() {
        let t = tournament();
        let flat = PickPopularity::chalk(&t, 1.0);
        let sharp = PickPopularity::chalk(&t, 2.5);
        assert!(
            sharp.favourite().1 > flat.favourite().1,
            "tilt did not concentrate: {:.3} vs {:.3}",
            sharp.favourite().1,
            flat.favourite().1
        );
    }

    #[test]
    fn a_popularity_file_round_trips_through_resolution() {
        let t = tournament();
        let p = chalk(&t);
        let file = PickPopularityFile {
            source: "test".into(),
            sample_size: Some(100),
            teams: t
                .teams
                .iter()
                .map(|team| TeamPicks {
                    name: team.name.clone(),
                    seed: team.seed,
                    region_id: t.region_rank[team.team_index as usize] as i32,
                    reach: (0..NUM_ROUNDS)
                        .map(|r| p.reach(r, team.team_index) as f64)
                        .collect(),
                })
                .collect(),
        };
        let restored = PickPopularity::from_file(&file, &t).unwrap();
        for round in 0..NUM_ROUNDS {
            for team in 0..NUM_TEAMS as u8 {
                assert!((restored.reach(round, team) - p.reach(round, team)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_file_missing_teams_is_an_error() {
        let t = tournament();
        let p = chalk(&t);
        let file = PickPopularityFile {
            source: "short".into(),
            sample_size: None,
            teams: t.teams[..60]
                .iter()
                .map(|team| TeamPicks {
                    name: team.name.clone(),
                    seed: team.seed,
                    region_id: t.region_rank[team.team_index as usize] as i32,
                    reach: (0..NUM_ROUNDS)
                        .map(|r| p.reach(r, team.team_index) as f64)
                        .collect(),
                })
                .collect(),
        };
        let err = PickPopularity::from_file(&file, &t).unwrap_err();
        assert!(err.contains("expected 64 teams of pick data, got 60"), "{}", err);
    }
}
