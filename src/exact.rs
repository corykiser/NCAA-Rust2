//! Exact optimal single bracket.
//!
//! Maximizing expected score does not need a genetic algorithm — it has a
//! closed form. A bracket's expected score is
//!
//! ```text
//! E[score] = SUM over games g of  points(round(g), seed(pick_g)) * P(pick_g wins g)
//! ```
//!
//! and `P(t wins g)` (see `crate::advancement`) depends only on the rating
//! model, never on your picks. The only coupling between games is structural:
//! the team you pick to win a game must be the team you picked to win one of
//! the two games feeding it. That makes the objective a dynamic program over
//! the bracket tree.
//!
//! Let `best[g][t]` be the most expected points obtainable from game `g` and
//! everything below it, given you pick `t` to win `g`:
//!
//! ```text
//! best[g][t] = points(round(g), seed(t)) * P(t wins g)
//!            + best[child containing t][t]
//!            + MAX over u of best[other child][u]
//! ```
//!
//! The optimum is `max_t best[62][t]`, recovered by walking back down. There
//! are 64 teams over 6 rounds, so the whole solve is a few thousand
//! multiply-adds — microseconds, deterministic, and provably optimal rather
//! than "the best of however many random brackets we sampled".
//!
//! Locks (`--lock-team`) drop straight into the same recurrence: forcing team
//! `t` to win game `g` just removes every other team from `g`'s candidate set.

use crate::bracket::{Bracket, ScoreTable, ScoringConfig};
use crate::ingest::TournamentInfo;
use crate::tree::{CHILDREN, NUM_GAMES, NUM_TEAMS, PARENT, ROUND_OF};

/// A requirement that a team win a given number of games.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TeamLock {
    pub team_index: u8,
    pub wins_required: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockError {
    /// Two locks demand different winners of the same game.
    Conflict {
        game: usize,
        round: usize,
        first: String,
        second: String,
    },
    /// More wins than the tournament has rounds.
    TooManyWins { team: String, wins: usize },
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::Conflict {
                round,
                first,
                second,
                ..
            } => write!(
                f,
                "{} and {} cannot both win their round-{} game — they meet there",
                first,
                second,
                round + 1
            ),
            LockError::TooManyWins { team, wins } => write!(
                f,
                "{} cannot win {} games; the tournament is only 6 rounds",
                team, wins
            ),
        }
    }
}

impl std::error::Error for LockError {}

/// The games each locked team must win, or an error if two locks collide.
///
/// A team that must win `n` games occupies one game per round along the single
/// path from its first-round slot to the final, so the mapping is just a walk
/// up the parent chain.
pub fn forced_winners(
    tournament: &TournamentInfo,
    locks: &[TeamLock],
) -> Result<[Option<u8>; NUM_GAMES], LockError> {
    let mut forced: [Option<u8>; NUM_GAMES] = [None; NUM_GAMES];

    for lock in locks {
        if lock.wins_required > 6 {
            return Err(LockError::TooManyWins {
                team: tournament.teams[lock.team_index as usize].name.clone(),
                wins: lock.wins_required,
            });
        }

        let mut game = tournament.r1_game_of_team[lock.team_index as usize];
        for _ in 0..lock.wins_required {
            match forced[game] {
                Some(other) if other != lock.team_index => {
                    return Err(LockError::Conflict {
                        game,
                        round: ROUND_OF[game],
                        first: tournament.teams[other as usize].name.clone(),
                        second: tournament.teams[lock.team_index as usize].name.clone(),
                    });
                }
                _ => forced[game] = Some(lock.team_index),
            }
            game = PARENT[game];
            if game == crate::tree::NO_GAME {
                break;
            }
        }
    }

    Ok(forced)
}

/// Result of an exact solve.
pub struct ExactSolution {
    pub bracket: Bracket,
    /// Expected score of `bracket`. This is the true optimum for the given
    /// scoring rules and locks — no other legal bracket scores higher.
    pub expected_value: f64,
}

/// Solve for the expected-value-maximizing bracket, subject to `locks`.
pub fn solve(
    tournament: &TournamentInfo,
    scoring: &ScoringConfig,
    locks: &[TeamLock],
) -> Result<ExactSolution, LockError> {
    let forced = forced_winners(tournament, locks)?;
    let table = ScoreTable::new(scoring);
    let adv = &tournament.advancement;

    // best[g][t]: expected points from g's subtree given t wins g.
    let mut best = vec![[f64::NEG_INFINITY; NUM_TEAMS]; NUM_GAMES];
    // Best achievable in each subtree, and the pick that achieves it.
    let mut best_sub = [f64::NEG_INFINITY; NUM_GAMES];
    let mut best_pick = [0u8; NUM_GAMES];

    // Games are numbered so that every child has a lower index than its parent,
    // which makes a single forward pass a valid topological order.
    for game in 0..NUM_GAMES {
        let round = ROUND_OF[game];

        for &t in adv.subtree(game) {
            if let Some(required) = forced[game] {
                if t != required {
                    continue;
                }
            }

            let own = table.get(round, tournament.seed_of[t as usize]) * adv.win_prob(game, t);

            let total = if game < 32 {
                own
            } else {
                let [c0, c1] = CHILDREN[game];
                let (near, far) = if adv.subtree_contains(c0, t) {
                    (c0, c1)
                } else {
                    (c1, c0)
                };
                // Either side may be unsatisfiable under the locks.
                if best[near][t as usize] == f64::NEG_INFINITY
                    || best_sub[far] == f64::NEG_INFINITY
                {
                    continue;
                }
                own + best[near][t as usize] + best_sub[far]
            };

            best[game][t as usize] = total;
            if total > best_sub[game] {
                best_sub[game] = total;
                best_pick[game] = t;
            }
        }

        if best_sub[game] == f64::NEG_INFINITY {
            // Only reachable if the locks are mutually unsatisfiable, which
            // forced_winners already rejects; guard anyway rather than emit a
            // silently wrong bracket.
            return Err(LockError::Conflict {
                game,
                round,
                first: "constraints".to_string(),
                second: "bracket structure".to_string(),
            });
        }
    }

    let mut winners = [0u8; NUM_GAMES];
    assign_winner(NUM_GAMES - 1, best_pick[NUM_GAMES - 1], adv, &best_pick, &mut winners);

    let binary = tournament.binary_from_winners(&winners);
    let bracket = Bracket::new_from_binary(tournament, &binary, Some(scoring));

    Ok(ExactSolution {
        expected_value: best_sub[NUM_GAMES - 1],
        bracket,
    })
}

fn assign_winner(
    game: usize,
    team: u8,
    adv: &crate::advancement::AdvancementModel,
    best_pick: &[u8; NUM_GAMES],
    winners: &mut [u8; NUM_GAMES],
) {
    winners[game] = team;
    if game < 32 {
        return;
    }
    let [c0, c1] = CHILDREN[game];
    let (near, far) = if adv.subtree_contains(c0, team) {
        (c0, c1)
    } else {
        (c1, c0)
    };
    assign_winner(near, team, adv, best_pick, winners);
    assign_winner(far, best_pick[far], adv, best_pick, winners);
}

/// Expected score of an arbitrary bracket under `scoring`.
///
/// Unlike a Monte Carlo estimate this has no sampling error, so two brackets
/// can be compared directly however small the difference between them.
pub fn expected_value(
    tournament: &TournamentInfo,
    bracket: &Bracket,
    scoring: &ScoringConfig,
) -> f64 {
    let table = ScoreTable::new(scoring);
    bracket
        .games
        .iter()
        .enumerate()
        .map(|(game, g)| {
            table.get(ROUND_OF[game], g.winner.seed)
                * tournament.advancement.win_prob(game, g.winner.team_index)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bracket::{ScoreTable, SeedScoring};
    use crate::ingest::tests::tournament;
    use crate::tree::{ROUND_GAMES, ROUND_START};

    /// Scoring that pays nothing for the Final Four or the final, which makes
    /// the four regions independent of each other.
    fn regions_only() -> ScoringConfig {
        let mut c = ScoringConfig::default();
        c.round_scores[4] = 0.0;
        c.round_scores[5] = 0.0;
        c.round_seed_scoring[4] = SeedScoring::None;
        c.round_seed_scoring[5] = SeedScoring::None;
        c
    }

    /// Best expected value obtainable in one region, by enumerating all 2^15
    /// ways that region can play out.
    fn brute_force_region(
        t: &crate::ingest::TournamentInfo,
        region: usize,
        scoring: &ScoringConfig,
    ) -> f64 {
        let table = ScoreTable::new(scoring);
        let games: Vec<usize> = (region * 8..region * 8 + 8)
            .chain(32 + region * 4..32 + region * 4 + 4)
            .chain(48 + region * 2..48 + region * 2 + 2)
            .chain(std::iter::once(56 + region))
            .collect();
        assert_eq!(games.len(), 15);

        let mut best = f64::NEG_INFINITY;

        for pattern in 0u32..(1 << 15) {
            let mut winners = [0u8; NUM_GAMES];
            let mut value = 0.0;

            for (bit, &game) in games.iter().enumerate() {
                let (a, b) = t.participants(game, &winners);
                let bit_true = t.bit_true_winner(a, b);
                let other = if bit_true == a { b } else { a };
                let w = if pattern >> bit & 1 == 1 { bit_true } else { other };
                winners[game] = w;
                value += table.get(ROUND_OF[game], t.seed_of[w as usize])
                    * t.advancement.win_prob(game, w);
            }

            if value > best {
                best = value;
            }
        }

        best
    }

    #[test]
    fn the_dynamic_program_matches_exhaustive_search() {
        // With the last two rounds worth nothing the regions decouple, so the
        // optimum over the whole bracket must equal the sum of the four region
        // optima found by brute force over every one of their 32,768 outcomes.
        let t = tournament();
        let scoring = regions_only();

        let solution = solve(&t, &scoring, &[]).unwrap();
        let brute: f64 = (0..4).map(|r| brute_force_region(&t, r, &scoring)).sum();

        assert!(
            (solution.expected_value - brute).abs() < 1e-9,
            "dp {:.9} vs exhaustive {:.9}",
            solution.expected_value,
            brute
        );
    }

    #[test]
    fn advancement_probabilities_are_a_distribution() {
        let t = tournament();
        for game in 0..NUM_GAMES {
            let total: f64 = t
                .advancement
                .subtree(game)
                .iter()
                .map(|&team| t.advancement.win_prob(game, team))
                .sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "game {} probabilities sum to {}",
                game,
                total
            );
        }
    }

    #[test]
    fn the_reported_optimum_is_the_brackets_own_expected_value() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let solution = solve(&t, &scoring, &[]).unwrap();
        let recomputed = expected_value(&t, &solution.bracket, &scoring);
        assert!((solution.expected_value - recomputed).abs() < 1e-9);
        assert!((solution.bracket.expected_value - recomputed).abs() < 1e-9);
        crate::bracket::tests::assert_legal(&solution.bracket);
    }

    #[test]
    fn no_neighbouring_bracket_beats_the_optimum() {
        use crate::ga::TeamRoundMutator;

        let t = tournament();
        let scoring = ScoringConfig::default();
        let solution = solve(&t, &scoring, &[]).unwrap();
        let best = solution.expected_value;

        // Every single-bit flip.
        for bit in 0..NUM_GAMES {
            let mut binary = solution.bracket.binary.clone();
            binary[bit] = !binary[bit];
            let candidate = crate::bracket::Bracket::new_from_binary(&t, &binary, Some(&scoring));
            assert!(
                candidate.expected_value <= best + 1e-9,
                "flipping bit {} improved on the optimum",
                bit
            );
        }

        // Every "force this team to this round" move.
        for team in 0..64u8 {
            for wins in 1..=6usize {
                let binary = TeamRoundMutator::force_index_to_round(
                    &solution.bracket.binary,
                    &t,
                    team,
                    wins,
                );
                let candidate =
                    crate::bracket::Bracket::new_from_binary(&t, &binary, Some(&scoring));
                assert!(
                    candidate.expected_value <= best + 1e-9,
                    "forcing team {} to win {} games improved on the optimum",
                    team,
                    wins
                );
            }
        }
    }

    #[test]
    fn solving_is_deterministic() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let first = solve(&t, &scoring, &[]).unwrap();
        for _ in 0..5 {
            let again = solve(&t, &scoring, &[]).unwrap();
            assert_eq!(again.bracket.binary, first.bracket.binary);
            assert_eq!(again.expected_value, first.expected_value);
        }
    }

    #[test]
    fn locks_are_honoured_and_cost_expected_value() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let unconstrained = solve(&t, &scoring, &[]).unwrap();

        // A 16 seed winning it all is the most expensive lock available.
        let long_shot = t
            .teams
            .iter()
            .find(|team| team.seed == 16)
            .unwrap()
            .team_index;
        let locked = solve(
            &t,
            &scoring,
            &[TeamLock {
                team_index: long_shot,
                wins_required: 6,
            }],
        )
        .unwrap();

        crate::bracket::tests::assert_legal(&locked.bracket);
        assert_eq!(locked.bracket.winner.team_index, long_shot);
        assert_eq!(locked.bracket.wins_for(long_shot), 6);
        assert!(locked.expected_value < unconstrained.expected_value);
    }

    #[test]
    fn a_lock_still_optimizes_everything_it_does_not_pin() {
        // Constraining one team must not disturb the rest of the bracket: games
        // in the other half of the draw should match the unconstrained optimum.
        let t = tournament();
        let scoring = ScoringConfig::default();
        let unconstrained = solve(&t, &scoring, &[]).unwrap();

        let pinned = t.teams.iter().find(|x| x.seed == 12).unwrap().team_index;
        let locked = solve(
            &t,
            &scoring,
            &[TeamLock {
                team_index: pinned,
                wins_required: 3,
            }],
        )
        .unwrap();

        // The Elite 8 game of a different region is untouched by the lock.
        let pinned_region = t.region_rank[pinned as usize];
        let mut compared = 0;
        for region in 0..4 {
            let game = 56 + region;
            let winner = unconstrained.bracket.games[game].winner.team_index;
            if t.region_rank[winner as usize] != pinned_region {
                assert_eq!(
                    locked.bracket.games[game].winner.team_index, winner,
                    "region {} changed despite the lock being elsewhere",
                    region
                );
                compared += 1;
            }
        }
        assert!(compared >= 2);
    }

    #[test]
    fn impossible_lock_pairs_are_rejected_before_any_work_happens() {
        let t = tournament();
        // Two teams from the same round-1 game cannot both win it.
        let game = t.r1_teams[0];
        let err = forced_winners(
            &t,
            &[
                TeamLock {
                    team_index: game[0],
                    wins_required: 1,
                },
                TeamLock {
                    team_index: game[1],
                    wins_required: 1,
                },
            ],
        )
        .unwrap_err();
        assert!(matches!(err, LockError::Conflict { .. }));

        // Two champions is likewise impossible.
        assert!(matches!(
            forced_winners(
                &t,
                &[
                    TeamLock { team_index: 0, wins_required: 6 },
                    TeamLock { team_index: 40, wins_required: 6 },
                ],
            ),
            Err(LockError::Conflict { .. })
        ));
    }

    #[test]
    fn every_round_is_scored_exactly_once() {
        let t = tournament();
        let scoring = ScoringConfig::default();
        let solution = solve(&t, &scoring, &[]).unwrap();
        let table = ScoreTable::new(&scoring);

        let mut total = 0.0;
        for round in 0..6 {
            for game in ROUND_START[round]..ROUND_START[round] + ROUND_GAMES[round] {
                let w = solution.bracket.games[game].winner.team_index;
                total += table.get(round, t.seed_of[w as usize]) * t.advancement.win_prob(game, w);
            }
        }
        assert!((total - solution.expected_value).abs() < 1e-9);
    }
}
