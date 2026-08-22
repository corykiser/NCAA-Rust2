//! Static structure of the 64-team bracket.
//!
//! Games are stored in a flat array of 63 slots:
//! R1 `0..32`, R2 `32..48`, Sweet 16 `48..56`, Elite 8 `56..60`,
//! Final Four `60..62`, Championship `62`.
//!
//! Within a region, the eight round-1 games are laid out in *seed* order
//! (`1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9`), which is not the same as
//! tree order — `1v16`'s winner plays `8v9`'s winner, not `2v15`'s. Every round
//! after the first is in true tree order. The child/parent tables below encode
//! that layout once so no other module has to rediscover it.

pub const NUM_GAMES: usize = 63;
pub const NUM_ROUNDS: usize = 6;
pub const NUM_TEAMS: usize = 64;

/// Index of the first game of each round.
pub const ROUND_START: [usize; NUM_ROUNDS] = [0, 32, 48, 56, 60, 62];
/// Number of games in each round.
pub const ROUND_GAMES: [usize; NUM_ROUNDS] = [32, 16, 8, 4, 2, 1];

/// Sentinel for "no such game" (round-1 games have no children, the final has no parent).
pub const NO_GAME: usize = usize::MAX;

/// Region order used by the flat game layout: East, West, South, Midwest.
pub const REGION_ORDER: [&str; 4] = ["East", "West", "South", "Midwest"];

/// Seed matchups of the eight round-1 games within a region, in layout order.
pub const R1_MATCHUPS: [[i32; 2]; 8] = [
    [1, 16],
    [2, 15],
    [3, 14],
    [4, 13],
    [5, 12],
    [6, 11],
    [7, 10],
    [8, 9],
];

/// Which round-1 slots feed each round-2 game within a region.
const R2_CHILD_SLOTS: [[usize; 2]; 4] = [[0, 7], [4, 3], [5, 2], [6, 1]];
/// Which round-2 slots feed each Sweet 16 game within a region.
const R3_CHILD_SLOTS: [[usize; 2]; 2] = [[0, 1], [2, 3]];

/// The two games feeding each game. Round-1 games have `[NO_GAME, NO_GAME]`.
pub const CHILDREN: [[usize; 2]; NUM_GAMES] = build_children();
/// The game each winner advances to. The championship has `NO_GAME`.
pub const PARENT: [usize; NUM_GAMES] = build_parent();
/// Round index (0-5) of each game.
pub const ROUND_OF: [usize; NUM_GAMES] = build_round_of();

const fn build_children() -> [[usize; 2]; NUM_GAMES] {
    let mut c = [[NO_GAME; 2]; NUM_GAMES];

    let mut region = 0;
    while region < 4 {
        // Round 2: four games per region.
        let mut j = 0;
        while j < 4 {
            let g = 32 + region * 4 + j;
            c[g][0] = region * 8 + R2_CHILD_SLOTS[j][0];
            c[g][1] = region * 8 + R2_CHILD_SLOTS[j][1];
            j += 1;
        }

        // Sweet 16: two games per region.
        let mut k = 0;
        while k < 2 {
            let g = 48 + region * 2 + k;
            c[g][0] = 32 + region * 4 + R3_CHILD_SLOTS[k][0];
            c[g][1] = 32 + region * 4 + R3_CHILD_SLOTS[k][1];
            k += 1;
        }

        // Elite 8: one game per region.
        c[56 + region][0] = 48 + region * 2;
        c[56 + region][1] = 48 + region * 2 + 1;

        region += 1;
    }

    // Final Four: South (2) vs Midwest (3), then East (0) vs West (1).
    c[60] = [58, 59];
    c[61] = [56, 57];
    // Championship.
    c[62] = [60, 61];

    c
}

const fn build_parent() -> [usize; NUM_GAMES] {
    let mut p = [NO_GAME; NUM_GAMES];
    let children = build_children();

    let mut g = 32;
    while g < NUM_GAMES {
        p[children[g][0]] = g;
        p[children[g][1]] = g;
        g += 1;
    }

    p
}

const fn build_round_of() -> [usize; NUM_GAMES] {
    let mut r = [0usize; NUM_GAMES];
    let mut round = 0;
    while round < NUM_ROUNDS {
        let mut i = 0;
        while i < ROUND_GAMES[round] {
            r[ROUND_START[round] + i] = round;
            i += 1;
        }
        round += 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_game_but_the_final_has_a_parent() {
        for g in 0..NUM_GAMES - 1 {
            assert_ne!(PARENT[g], NO_GAME, "game {} has no parent", g);
        }
        assert_eq!(PARENT[62], NO_GAME);
    }

    #[test]
    fn children_are_consistent_with_parents() {
        for g in 32..NUM_GAMES {
            for &c in &CHILDREN[g] {
                assert_ne!(c, NO_GAME);
                assert_eq!(PARENT[c], g);
                assert_eq!(ROUND_OF[c] + 1, ROUND_OF[g]);
            }
        }
        for g in 0..32 {
            assert_eq!(CHILDREN[g], [NO_GAME, NO_GAME]);
        }
    }

    #[test]
    fn each_game_is_a_child_exactly_once() {
        let mut seen = [0usize; NUM_GAMES];
        for g in 32..NUM_GAMES {
            for &c in &CHILDREN[g] {
                seen[c] += 1;
            }
        }
        for g in 0..NUM_GAMES - 1 {
            assert_eq!(seen[g], 1, "game {} appears {} times as a child", g, seen[g]);
        }
        assert_eq!(seen[62], 0);
    }

    #[test]
    fn subtree_of_each_game_has_the_expected_size() {
        fn leaves(g: usize) -> usize {
            if CHILDREN[g][0] == NO_GAME {
                2
            } else {
                leaves(CHILDREN[g][0]) + leaves(CHILDREN[g][1])
            }
        }
        assert_eq!(leaves(62), NUM_TEAMS);
        for g in 0..NUM_GAMES {
            assert_eq!(leaves(g), 1 << (ROUND_OF[g] + 1));
        }
    }

    #[test]
    fn round1_seed_matchups_sum_to_seventeen() {
        for m in &R1_MATCHUPS {
            assert_eq!(m[0] + m[1], 17);
        }
    }

    #[test]
    fn top_seed_meets_the_eight_nine_winner_in_round_two() {
        // 1v16 is slot 0, 8v9 is slot 7; they must share a round-2 game.
        assert_eq!(CHILDREN[32], [0, 7]);
    }
}
