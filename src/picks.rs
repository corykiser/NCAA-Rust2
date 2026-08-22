//! A bracket as the optimizers see it.
//!
//! [`Bracket`](crate::bracket::Bracket) carries three `Arc<Team>` handles and a
//! probability triple per game — 63 `Game` structs, 189 atomic refcount bumps
//! and two heap allocations every time one is built. That is the right shape
//! for printing a result and the wrong shape for an inner loop that builds
//! millions of candidates, where the only things ever read are *who won each
//! game* and *what the encoding bit says*.
//!
//! `Picks` is that pair, in 72 bytes of `Copy` data with no allocation and no
//! reference counting. Every optimizer works on `Picks` and materialises a full
//! `Bracket` once, at the end, for display.
//!
//! The 64th winner slot is padding so the scoring kernel in [`crate::score`]
//! can work in whole 8-lane groups without a tail.

use crate::ingest::TournamentInfo;
use crate::tree::{CHILDREN, NO_GAME, NUM_GAMES, NUM_ROUNDS, NUM_TEAMS, PARENT};
use rand::Rng;

/// Winner slots per bracket: 63 games plus one padding lane.
pub const PADDED_GAMES: usize = 64;

/// Filler for the padding lane. No real team index can collide with it, so a
/// padded slot never counts as a match however the kernel is arranged.
const PAD: u8 = u8::MAX;

/// The 63 game winners of one bracket, plus its 63-bit encoding.
///
/// The two representations are kept in step by construction: every operation
/// that changes a winner also updates the corresponding bit, so neither ever
/// has to be re-derived from the other.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Picks {
    winners: [u8; PADDED_GAMES],
    bits: u64,
}

impl std::fmt::Debug for Picks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Picks(champion={}, bits={:#018x})", self.champion(), self.bits)
    }
}

impl Picks {
    /// Decode the 63-bit representation.
    pub fn from_bits(t: &TournamentInfo, bits: u64) -> Picks {
        let mut p = Picks {
            winners: [PAD; PADDED_GAMES],
            bits,
        };
        for game in 0..NUM_GAMES {
            let (a, b) = p.participants(t, game);
            p.winners[game] = p.decide(t, game, a, b);
        }
        p
    }

    /// Build from a decided winner for every game, deriving the bits.
    ///
    /// Panics in debug builds if `winners` is not a legal bracket.
    pub fn from_winners(t: &TournamentInfo, winners: &[u8; NUM_GAMES]) -> Picks {
        let mut p = Picks {
            winners: [PAD; PADDED_GAMES],
            bits: 0,
        };
        p.winners[..NUM_GAMES].copy_from_slice(winners);
        for game in 0..NUM_GAMES {
            let (a, b) = p.participants(t, game);
            let w = winners[game];
            debug_assert!(
                w == a || w == b,
                "game {} winner {} played neither {} nor {}",
                game,
                w,
                a,
                b
            );
            let loser = if w == a { b } else { a };
            p.set_bit(game, t.winner_bit(w, loser));
        }
        p
    }

    /// Sample one tournament outcome, each game decided by its win probability.
    ///
    /// Draws a single `u32` per game and compares it against a precomputed
    /// threshold, so a whole scenario costs 63 integer compares.
    pub fn sample(t: &TournamentInfo, rng: &mut impl Rng) -> Picks {
        let mut p = Picks {
            winners: [PAD; PADDED_GAMES],
            bits: 0,
        };
        for game in 0..NUM_GAMES {
            let (a, b) = p.participants(t, game);
            let w = if rng.gen::<u32>() < t.win_threshold[a as usize][b as usize] {
                a
            } else {
                b
            };
            let loser = if w == a { b } else { a };
            p.winners[game] = w;
            p.set_bit(game, t.winner_bit(w, loser));
        }
        p
    }

    /// The two teams playing `game`, given the winners already decided below it.
    #[inline(always)]
    fn participants(&self, t: &TournamentInfo, game: usize) -> (u8, u8) {
        if game < 32 {
            (t.r1_teams[game][0], t.r1_teams[game][1])
        } else {
            let [c0, c1] = CHILDREN[game];
            (self.winners[c0], self.winners[c1])
        }
    }

    /// Resolve `game`'s winner from its retained bit.
    #[inline(always)]
    fn decide(&self, t: &TournamentInfo, game: usize, a: u8, b: u8) -> u8 {
        let bit_true = t.bit_true_winner(a, b);
        if self.bit(game) {
            bit_true
        } else if bit_true == a {
            b
        } else {
            a
        }
    }

    #[inline(always)]
    fn set_bit(&mut self, game: usize, value: bool) {
        let mask = 1u64 << game;
        if value {
            self.bits |= mask;
        } else {
            self.bits &= !mask;
        }
    }

    /// The encoding bit of `game`.
    #[inline(always)]
    pub fn bit(&self, game: usize) -> bool {
        self.bits >> game & 1 != 0
    }

    /// The 63-bit encoding.
    #[inline(always)]
    pub fn bits(&self) -> u64 {
        self.bits
    }

    /// Winner of `game`.
    #[inline(always)]
    pub fn winner(&self, game: usize) -> u8 {
        self.winners[game]
    }

    /// Winners of all 63 games, followed by the padding lane.
    #[inline(always)]
    pub fn winner_lanes(&self) -> &[u8; PADDED_GAMES] {
        &self.winners
    }

    /// Winners of all 63 games.
    #[inline(always)]
    pub fn winners(&self) -> &[u8; NUM_GAMES] {
        // The first 63 lanes are exactly the game winners.
        self.winners[..NUM_GAMES].try_into().expect("63 of 64 lanes")
    }

    /// The team that wins the final.
    #[inline(always)]
    pub fn champion(&self) -> u8 {
        self.winners[NUM_GAMES - 1]
    }

    /// How many games `team` wins in this bracket.
    pub fn wins_for(&self, team: u8) -> usize {
        self.winners[..NUM_GAMES].iter().filter(|&&w| w == team).count()
    }

    /// Number of games where the two brackets pick different winners.
    pub fn disagreements(&self, other: &Picks) -> usize {
        (0..NUM_GAMES)
            .filter(|&g| self.winners[g] != other.winners[g])
            .count()
    }

    /// Rewrite the bracket so `team` wins its first `wins` games.
    ///
    /// Walks the single path from the team's round-1 slot up the parent chain,
    /// setting each bit against the opponent the already-rewritten games below
    /// actually deliver, then re-resolves the ancestors above the forced prefix
    /// from their retained bits. Games off that path are untouched, so the
    /// result is always a legal bracket and the work is 12 steps rather than a
    /// full 63-game decode.
    pub fn force_to_round(&mut self, t: &TournamentInfo, team: u8, wins: usize) {
        let mut game = t.r1_game_of_team[team as usize];

        for _ in 0..wins.min(NUM_ROUNDS) {
            let (a, b) = self.participants(t, game);
            let opponent = if a == team {
                b
            } else if b == team {
                a
            } else {
                debug_assert!(false, "team {} is not a participant of game {}", team, game);
                break;
            };

            self.winners[game] = team;
            self.set_bit(game, t.winner_bit(team, opponent));

            match PARENT[game] {
                NO_GAME => return,
                parent => game = parent,
            }
        }

        // `game` is now the lowest ancestor that kept its own bit; its
        // participants may have changed, and so may every ancestor above it.
        loop {
            let (a, b) = self.participants(t, game);
            self.winners[game] = self.decide(t, game, a, b);
            match PARENT[game] {
                NO_GAME => return,
                parent => game = parent,
            }
        }
    }

    /// `force_to_round` applied to a copy.
    pub fn forced_to_round(&self, t: &TournamentInfo, team: u8, wins: usize) -> Picks {
        let mut copy = *self;
        copy.force_to_round(t, team, wins);
        copy
    }

}

/// Every legal (team, wins) move, in a fixed order.
///
/// This is the neighbourhood the local searches explore: 64 teams times the six
/// depths each could be pushed to. Round-1 losses are not a separate move —
/// forcing a *different* team to win that game covers them.
pub fn all_moves() -> impl Iterator<Item = (u8, usize)> {
    (0..NUM_TEAMS as u8).flat_map(|team| (1..=NUM_ROUNDS).map(move |wins| (team, wins)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::tests::tournament;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn assert_legal(t: &TournamentInfo, p: &Picks) {
        for game in 32..NUM_GAMES {
            let [c0, c1] = CHILDREN[game];
            let w = p.winner(game);
            assert!(
                w == p.winner(c0) || w == p.winner(c1),
                "game {} winner {} is neither feeder winner ({}, {})",
                game,
                w,
                p.winner(c0),
                p.winner(c1)
            );
        }
        for game in 0..32 {
            let [a, b] = t.r1_teams[game];
            assert!(p.winner(game) == a || p.winner(game) == b);
        }
    }

    #[test]
    fn bits_and_winners_stay_in_step_through_every_constructor() {
        let t = tournament();
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        for _ in 0..200 {
            let sampled = Picks::sample(&t, &mut rng);
            assert_legal(&t, &sampled);
            // Both directions must reproduce the same object.
            assert_eq!(Picks::from_bits(&t, sampled.bits()), sampled);
            assert_eq!(Picks::from_winners(&t, sampled.winners()), sampled);
        }
    }

    #[test]
    fn forcing_a_team_to_a_round_wins_that_many_games() {
        let t = tournament();
        let mut rng = SmallRng::seed_from_u64(7);
        let start = Picks::sample(&t, &mut rng);
        for team in 0..64u8 {
            for wins in 1..=6usize {
                let p = start.forced_to_round(&t, team, wins);
                assert_legal(&t, &p);
                assert!(
                    p.wins_for(team) >= wins,
                    "team {} won {} of {} forced games",
                    team,
                    p.wins_for(team),
                    wins
                );
                if wins == 6 {
                    assert_eq!(p.champion(), team);
                }
                // The incremental rewrite must agree with a full re-decode.
                assert_eq!(Picks::from_bits(&t, p.bits()), p);
            }
        }
    }

    #[test]
    fn forcing_only_touches_the_teams_own_path() {
        let t = tournament();
        let mut rng = SmallRng::seed_from_u64(11);
        let start = Picks::sample(&t, &mut rng);
        let team = t.r1_teams[0][0];
        let forced = start.forced_to_round(&t, team, 4);

        let mut path = vec![t.r1_game_of_team[team as usize]];
        while path.len() < 4 {
            match PARENT[*path.last().unwrap()] {
                NO_GAME => break,
                p => path.push(p),
            }
        }
        for game in 0..NUM_GAMES {
            if !path.contains(&game) {
                assert_eq!(
                    forced.bit(game),
                    start.bit(game),
                    "game {} bit changed but is off the forced path",
                    game
                );
            }
        }
    }

    #[test]
    fn forcing_is_idempotent() {
        let t = tournament();
        let mut rng = SmallRng::seed_from_u64(17);
        let start = Picks::sample(&t, &mut rng);
        let once = start.forced_to_round(&t, 20, 4);
        assert_eq!(once.forced_to_round(&t, 20, 4), once);
    }

    #[test]
    fn the_move_list_covers_every_team_and_depth() {
        let moves: Vec<_> = all_moves().collect();
        assert_eq!(moves.len(), NUM_TEAMS * NUM_ROUNDS);
        assert_eq!(moves[0], (0, 1));
        assert_eq!(*moves.last().unwrap(), (63, 6));
    }
}
