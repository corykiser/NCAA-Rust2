//! Exact advancement probabilities for every team in every game.
//!
//! `alive[g][t]` is the unconditional probability that team `t` wins game `g`
//! — that is, the probability the tournament actually plays out such that `t`
//! emerges from that slot. It is computed once from the rating model and is
//! independent of anyone's picks, which is what makes exact expected-value
//! scoring possible (see `crate::exact`).
//!
//! The recurrence is the obvious one. For a round-1 game between `a` and `b`,
//! `alive[g][a] = P(a beats b)`. For any later game with children `c0`, `c1`:
//!
//! ```text
//! alive[g][t] = alive[c0][t] * SUM over u in subtree(c1) of alive[c1][u] * P(t beats u)
//! ```
//!
//! Total work is about 2,000 multiply-adds, so this is microseconds, not a
//! Monte Carlo estimate.

use crate::ingest::ProbabilityCache;
use crate::tree::{CHILDREN, NUM_GAMES, NUM_TEAMS};

#[derive(Debug, Clone)]
pub struct AdvancementModel {
    /// `alive[game][team_index]` = P(team wins that game). Zero for teams that
    /// cannot reach the game at all.
    alive: Vec<[f64; NUM_TEAMS]>,
    /// Team indices that can reach each game, in bracket order.
    subtree: Vec<Vec<u8>>,
    /// Bitset form of `subtree`, for O(1) "can this team reach this game" tests.
    subtree_mask: Vec<u64>,
}

impl AdvancementModel {
    /// Build the model from round-1 matchups and the pairwise win-probability cache.
    pub fn new(r1_teams: &[[u8; 2]; 32], cache: &ProbabilityCache) -> Self {
        let mut alive = vec![[0.0f64; NUM_TEAMS]; NUM_GAMES];
        let mut subtree: Vec<Vec<u8>> = vec![Vec::new(); NUM_GAMES];

        for (g, participants) in r1_teams.iter().enumerate() {
            let (a, b) = (participants[0], participants[1]);
            let p = cache.get(a, b);
            alive[g][a as usize] = p;
            alive[g][b as usize] = 1.0 - p;
            subtree[g] = vec![a, b];
        }

        for g in 32..NUM_GAMES {
            let [c0, c1] = CHILDREN[g];

            for (near, far) in [(c0, c1), (c1, c0)] {
                for &t in &subtree[near] {
                    let reach = alive[near][t as usize];
                    if reach == 0.0 {
                        continue;
                    }
                    let mut beats_field = 0.0;
                    for &u in &subtree[far] {
                        beats_field += alive[far][u as usize] * cache.get(t, u);
                    }
                    alive[g][t as usize] = reach * beats_field;
                }
            }

            let mut teams = subtree[c0].clone();
            teams.extend_from_slice(&subtree[c1]);
            subtree[g] = teams;
        }

        let subtree_mask = subtree
            .iter()
            .map(|teams| teams.iter().fold(0u64, |mask, &t| mask | (1u64 << t)))
            .collect();

        AdvancementModel {
            alive,
            subtree,
            subtree_mask,
        }
    }

    /// Probability that `team` wins `game`.
    #[inline]
    pub fn win_prob(&self, game: usize, team: u8) -> f64 {
        self.alive[game][team as usize]
    }

    /// Every team that can reach `game`, in bracket order.
    #[inline]
    pub fn subtree(&self, game: usize) -> &[u8] {
        &self.subtree[game]
    }

    /// Whether `team` can reach `game` at all.
    #[inline]
    pub fn subtree_contains(&self, game: usize, team: u8) -> bool {
        self.subtree_mask[game] & (1u64 << team) != 0
    }
}
