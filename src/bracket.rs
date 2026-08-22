use crate::ingest::{RcTeam, TournamentInfo};
use crate::picks::Picks;
use crate::tree::{NUM_GAMES, ROUND_GAMES, ROUND_OF, ROUND_START};
use rand::Rng;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeedScoring {
    Add,
    Multiply,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoringConfig {
    pub round_scores: [f64; 6],
    pub round_seed_scoring: [SeedScoring; 6],
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            round_scores: [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            round_seed_scoring: [
                SeedScoring::Add,      // R1: 1 + seed
                SeedScoring::Add,      // R2: 2 + seed
                SeedScoring::Add,      // R3: 4 + seed
                SeedScoring::Multiply, // R4: 8 * seed
                SeedScoring::Multiply, // R5: 16 * seed
                SeedScoring::Multiply, // R6: 32 * seed
            ],
        }
    }
}

/// Pre-computed score lookup table for fast scoring.
/// Avoids recomputing the seed adjustment on every scored pick.
#[derive(Debug, Clone, Copy)]
pub struct ScoreTable {
    /// scores[round][seed] = points for a correct pick. Seeds are 1-16;
    /// index 0 is unused so seeds index directly.
    scores: [[f64; 17]; 6],
}

impl ScoreTable {
    pub fn new(config: &ScoringConfig) -> Self {
        let mut scores = [[0.0; 17]; 6];

        for round in 0..6 {
            let base_score = config.round_scores[round];
            for seed in 1..=16 {
                scores[round][seed] = match config.round_seed_scoring[round] {
                    SeedScoring::Add => base_score + seed as f64,
                    SeedScoring::Multiply => base_score * seed as f64,
                    SeedScoring::None => base_score,
                };
            }
        }

        ScoreTable { scores }
    }

    /// Points for a correct pick.
    ///
    /// Seeds are validated to be 1-16 when the tournament field is built, so
    /// this is always in range; the bounds check costs nothing next to the
    /// memory traffic in the scoring loop and is worth keeping over the
    /// `get_unchecked` this used to do on a value parsed from a data file.
    #[inline(always)]
    pub fn get(&self, round: usize, seed: i32) -> f64 {
        self.scores[round][seed as usize]
    }
}

impl Default for ScoreTable {
    fn default() -> Self {
        Self::new(&ScoringConfig::default())
    }
}

/// Bracket reduced to what scoring actually reads: the winning team of each
/// game, plus that team's seed. Used for the hot comparison loop.
#[derive(Debug, Clone)]
pub struct FastBracket {
    /// Winner team index (0-63) for each of 63 games
    pub winners: [u8; NUM_GAMES],
    /// Winner seed for each game (for score calculation)
    pub winner_seeds: [i32; NUM_GAMES],
}

impl FastBracket {
    pub fn from_bracket(bracket: &Bracket) -> Self {
        let mut winners = [0u8; NUM_GAMES];
        let mut winner_seeds = [0i32; NUM_GAMES];

        for (i, game) in bracket.games.iter().enumerate() {
            winners[i] = game.winner.team_index;
            winner_seeds[i] = game.winner.seed;
        }

        FastBracket {
            winners,
            winner_seeds,
        }
    }

    /// Score this bracket against an outcome using the pre-computed table.
    #[inline]
    pub fn score_against(&self, other: &FastBracket, table: &ScoreTable) -> f64 {
        let mut score: f64 = 0.0;
        for i in 0..NUM_GAMES {
            if self.winners[i] == other.winners[i] {
                score += table.get(ROUND_OF[i], self.winner_seeds[i]);
            }
        }
        score
    }
}

/// A single game: who played, how likely each was to win it, and who did.
#[derive(Debug, Clone)]
pub struct Game {
    pub team1: RcTeam,
    pub team2: RcTeam,
    pub team1prob: f64,
    pub team2prob: f64,
    pub winnerprob: f64,
    pub winner: RcTeam,
    /// Encoding bit: did the lower seed win (or, across regions, the
    /// alphabetically earlier region)? See `TournamentInfo::bit_true_winner`.
    pub lower_seed_won: bool,
}

/// Result of comparing two brackets with weighted distance metric
#[derive(Debug, Clone)]
pub struct BracketDistance {
    /// Total weighted distance (lower = more similar)
    pub total_distance: f64,
    /// Similarity percentage (0.0 to 1.0, higher = more similar)
    pub similarity: f64,
    /// Distance contribution from each round [R1, R2, Sweet16, Elite8, F4, Championship]
    pub round_distances: [f64; 6],
    /// Number of matching picks per round
    pub round_matches: [usize; 6],
    /// Total games per round for reference
    pub round_totals: [usize; 6],
    /// Whether both brackets have the same champion
    pub champion_match: bool,
}

impl BracketDistance {
    pub fn print(&self) {
        let round_names = [
            "Round 1",
            "Round 2",
            "Sweet 16",
            "Elite 8",
            "Final Four",
            "Championship",
        ];

        println!("\nBracket Similarity Analysis");
        println!("============================");
        println!("Overall Similarity: {:.1}%", self.similarity * 100.0);
        println!("Total Distance: {:.2}", self.total_distance);
        println!(
            "Same Champion: {}",
            if self.champion_match { "Yes" } else { "No" }
        );
        println!();
        println!("{:<14} {:>8} {:>12}", "Round", "Matches", "Distance");
        println!("{}", "-".repeat(36));

        for i in 0..6 {
            println!(
                "{:<14} {:>3}/{:<4} {:>12.2}",
                round_names[i], self.round_matches[i], self.round_totals[i], self.round_distances[i]
            );
        }
        println!();
    }

    pub fn is_identical(&self) -> bool {
        self.total_distance == 0.0
    }

    /// Percentage of games that match from the Sweet 16 on.
    pub fn late_round_match_pct(&self) -> f64 {
        let late_matches: usize = self.round_matches[2..].iter().sum();
        let late_totals: usize = self.round_totals[2..].iter().sum();
        late_matches as f64 / late_totals as f64
    }
}

#[derive(Debug, Clone)]
pub struct Bracket {
    /// 63 games. R1 0-31, R2 32-47, S16 48-55, E8 56-59, F4 60-61, final 62.
    pub games: Vec<Game>,
    pub winner: RcTeam,
    /// Probability this exact set of 63 outcomes occurs.
    pub prob: f64,
    /// Score this bracket would earn if every pick came in.
    pub score: f64,
    pub sim_score: f64,
    /// Expected score: the sum over games of `points * P(pick actually wins that game)`.
    /// The probability is unconditional — it already accounts for the pick having
    /// to survive earlier rounds — so this is directly comparable to a mean score
    /// against simulated tournaments.
    pub expected_value: f64,
    pub binary: Vec<bool>,
}

impl PartialEq for Bracket {
    fn eq(&self, other: &Self) -> bool {
        self.binary == other.binary
    }
}

impl Bracket {
    /// Build the full bracket from a decided winner for every game.
    ///
    /// Every constructor funnels through here, so the tree layout lives in
    /// exactly one place (`crate::tree`) instead of being re-derived per round.
    fn assemble(
        tournamentinfo: &TournamentInfo,
        winners: &[u8; NUM_GAMES],
        config: &ScoringConfig,
    ) -> Bracket {
        Self::assemble_with(tournamentinfo, winners, &ScoreTable::new(config))
    }

    /// `assemble` against an already-built score table.
    ///
    /// Rebuilding the 6x17 table per bracket was 102 divisions and branches
    /// before a single game was looked at; callers that construct brackets in
    /// bulk build it once.
    fn assemble_with(
        tournamentinfo: &TournamentInfo,
        winners: &[u8; NUM_GAMES],
        table: &ScoreTable,
    ) -> Bracket {
        let mut games: Vec<Game> = Vec::with_capacity(NUM_GAMES);
        let mut binary: Vec<bool> = Vec::with_capacity(NUM_GAMES);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value: f64 = 0.0;

        for game_idx in 0..NUM_GAMES {
            let (a, b) = tournamentinfo.participants(game_idx, winners);
            let w = winners[game_idx];
            debug_assert!(w == a || w == b);
            let loser = if w == a { b } else { a };

            let team1prob = tournamentinfo.prob_cache.get(a, b);
            let team2prob = 1.0 - team1prob;
            let winnerprob = if w == a { team1prob } else { team2prob };

            let winner = Arc::clone(&tournamentinfo.teams[w as usize]);
            let win_score = table.get(ROUND_OF[game_idx], winner.seed);

            prob *= winnerprob;
            score += win_score;
            expected_value += win_score * tournamentinfo.advancement.win_prob(game_idx, w);

            let lower_seed_won = tournamentinfo.winner_bit(w, loser);
            binary.push(lower_seed_won);

            games.push(Game {
                team1: Arc::clone(&tournamentinfo.teams[a as usize]),
                team2: Arc::clone(&tournamentinfo.teams[b as usize]),
                team1prob,
                team2prob,
                winnerprob,
                winner,
                lower_seed_won,
            });
        }

        let tournament_winner = Arc::clone(&games[NUM_GAMES - 1].winner);

        Bracket {
            games,
            winner: tournament_winner,
            prob,
            score,
            sim_score: 0.0,
            expected_value,
            binary,
        }
    }

    /// Sample one tournament outcome, each game decided by its win probability.
    pub fn new(tournamentinfo: &TournamentInfo, config: Option<&ScoringConfig>) -> Bracket {
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);

        let mut rng = rand::thread_rng();
        let mut winners = [0u8; NUM_GAMES];

        for game_idx in 0..NUM_GAMES {
            let (a, b) = tournamentinfo.participants(game_idx, &winners);
            let p = tournamentinfo.prob_cache.get(a, b);
            winners[game_idx] = if rng.gen::<f64>() < p { a } else { b };
        }

        Self::assemble(tournamentinfo, &winners, config)
    }

    /// Build a bracket from its 63-bit representation.
    pub fn new_from_binary(
        tournamentinfo: &TournamentInfo,
        binary_slice: &[bool],
        config: Option<&ScoringConfig>,
    ) -> Bracket {
        assert!(
            binary_slice.len() == NUM_GAMES,
            "Binary slice must be {} elements long",
            NUM_GAMES
        );
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);

        let winners = tournamentinfo.decode_winners(binary_slice);
        Self::assemble(tournamentinfo, &winners, config)
    }

    /// Build a bracket from its 63-bit `u64` encoding.
    pub fn new_from_binary_bits(
        tournamentinfo: &TournamentInfo,
        bits: u64,
        config: Option<&ScoringConfig>,
    ) -> Bracket {
        Self::from_picks(
            tournamentinfo,
            &Picks::from_bits(tournamentinfo, bits),
            config,
        )
    }

    /// Materialise the full display bracket from the optimizers' compact form.
    ///
    /// This is the one place the two representations meet: everything that
    /// searches works on `Picks`, and exactly one `Bracket` per reported result
    /// is built here.
    pub fn from_picks(
        tournamentinfo: &TournamentInfo,
        picks: &Picks,
        config: Option<&ScoringConfig>,
    ) -> Bracket {
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);
        Self::assemble(tournamentinfo, picks.winners(), config)
    }

    /// `from_picks` against an already-built score table.
    pub fn from_picks_with(
        tournamentinfo: &TournamentInfo,
        picks: &Picks,
        table: &ScoreTable,
    ) -> Bracket {
        Self::assemble_with(tournamentinfo, picks.winners(), table)
    }

    /// The compact form the optimizers work on.
    pub fn picks(&self, tournamentinfo: &TournamentInfo) -> Picks {
        Picks::from_winners(tournamentinfo, &self.winner_indices())
    }

    /// Winning team index of every game.
    pub fn winner_indices(&self) -> [u8; NUM_GAMES] {
        let mut winners = [0u8; NUM_GAMES];
        for (i, game) in self.games.iter().enumerate() {
            winners[i] = game.winner.team_index;
        }
        winners
    }

    /// Number of games this team wins in this bracket.
    pub fn wins_for(&self, team_index: u8) -> usize {
        self.games
            .iter()
            .filter(|g| g.winner.team_index == team_index)
            .count()
    }

    pub fn score(&self, referencebracket: &Bracket, config: Option<&ScoringConfig>) -> f64 {
        let default_config = ScoringConfig::default();
        let table = ScoreTable::new(config.unwrap_or(&default_config));
        self.score_fast(referencebracket, &table)
    }

    /// Score against a reference outcome using a pre-computed table.
    #[inline]
    pub fn score_fast(&self, referencebracket: &Bracket, table: &ScoreTable) -> f64 {
        let mut score: f64 = 0.0;
        for i in 0..NUM_GAMES {
            if Arc::ptr_eq(&self.games[i].winner, &referencebracket.games[i].winner) {
                score += table.get(ROUND_OF[i], self.games[i].winner.seed);
            }
        }
        score
    }

    /// Create a mutated copy by flipping bits at random.
    ///
    /// Note that a bit's *meaning* depends on who reaches that game, so an
    /// early flip changes what later bits refer to. `TeamRoundMutator` is the
    /// directed alternative and is what the optimizers use.
    pub fn mutate(
        &self,
        tournamentinfo: &TournamentInfo,
        mutation_rate: f64,
        config: Option<&ScoringConfig>,
    ) -> Bracket {
        let mut new_binary: Vec<bool> = self.binary.clone();
        let mut rng = rand::thread_rng();
        for bit in new_binary.iter_mut() {
            let rand: f64 = rng.gen();
            if rand < mutation_rate {
                *bit = !*bit;
            }
        }
        Bracket::new_from_binary(tournamentinfo, &new_binary, config)
    }

    pub fn create_n_children(
        &mut self,
        tournamentinfo: &TournamentInfo,
        n: usize,
        mutation_rate: f64,
        config: Option<&ScoringConfig>,
    ) -> Vec<Bracket> {
        (0..n)
            .map(|_| self.mutate(tournamentinfo, mutation_rate, config))
            .collect()
    }

    pub fn hamming_distance(&self, other: &Bracket) -> usize {
        self.binary
            .iter()
            .zip(other.binary.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Weighted distance metric that values later rounds more heavily
    pub fn weighted_distance(&self, other: &Bracket, config: Option<&ScoringConfig>) -> BracketDistance {
        let default_config = ScoringConfig::default();
        let config = config.unwrap_or(&default_config);
        let weights = config.round_scores;

        let mut round_distances: [f64; 6] = [0.0; 6];
        let mut round_matches: [usize; 6] = [0; 6];
        let mut round_totals: [usize; 6] = [0; 6];
        let mut max_distance = 0.0;

        for round in 0..6 {
            round_totals[round] = ROUND_GAMES[round];
            max_distance += ROUND_GAMES[round] as f64 * weights[round] * 2.0;

            for i in ROUND_START[round]..ROUND_START[round] + ROUND_GAMES[round] {
                if self.games[i].winner == other.games[i].winner {
                    round_matches[round] += 1;
                } else {
                    let seed_weight =
                        (self.games[i].winner.seed + other.games[i].winner.seed) as f64 / 2.0;
                    round_distances[round] += weights[round] * (1.0 + seed_weight / 16.0);
                }
            }
        }

        let total_distance: f64 = round_distances.iter().sum();

        BracketDistance {
            total_distance,
            similarity: 1.0 - (total_distance / max_distance),
            round_distances,
            round_matches,
            round_totals,
            champion_match: self.winner == other.winner,
        }
    }

    /// Print the bracket, one section per round.
    ///
    /// Each heading names the round the listed teams *advance to*, matching the
    /// vocabulary `--lock-team` uses. The old headings named the round just
    /// played, so `--lock-team X:FinalFour` put X under a heading reading
    /// "Elite 8" and looked as though the lock had been ignored.
    pub fn pretty_print(&self) {
        let advances_to = [
            "Round of 32",
            "Sweet 16",
            "Elite Eight",
            "Final Four",
            "Championship Game",
            "Champion",
        ];

        for round in 0..6 {
            println!("{}", advances_to[round]);
            let start = ROUND_START[round];
            for game in &self.games[start..start + ROUND_GAMES[round]] {
                if round == 5 {
                    println!("{} {} wins!", game.winner.seed, game.winner.name);
                } else {
                    println!("{} {}", game.winner.seed, game.winner.name);
                }
            }
            println!();
        }

        println!("Expected score: {:.2}", self.expected_value);
        println!("Maximum possible score: {:.0}", self.score);
        println!();
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::ingest::tests::tournament;
    use crate::tree::CHILDREN;

    /// The invariant that makes a bracket a bracket: the winner of a game is
    /// one of the two teams that played in it, and the teams that play in a
    /// game are the winners of the two games feeding it.
    pub fn assert_legal(bracket: &Bracket) {
        for (i, game) in bracket.games.iter().enumerate() {
            assert!(
                game.winner.team_index == game.team1.team_index
                    || game.winner.team_index == game.team2.team_index,
                "game {}: winner {} played neither {} nor {}",
                i,
                game.winner.name,
                game.team1.name,
                game.team2.name
            );

            if i >= 32 {
                let [c0, c1] = CHILDREN[i];
                let feeders = [
                    bracket.games[c0].winner.team_index,
                    bracket.games[c1].winner.team_index,
                ];
                let participants = [game.team1.team_index, game.team2.team_index];
                assert_eq!(
                    feeders, participants,
                    "game {}: participants {:?} are not the winners {:?} of games {} and {}",
                    i, participants, feeders, c0, c1
                );
            }
        }

        assert_eq!(bracket.winner.team_index, bracket.games[62].winner.team_index);
        assert_eq!(bracket.binary.len(), NUM_GAMES);
    }

    #[test]
    fn randomly_generated_brackets_are_legal() {
        let t = tournament();
        for _ in 0..50 {
            assert_legal(&Bracket::new(&t, None));
        }
    }

    #[test]
    fn brackets_built_from_bits_are_legal_and_round_trip() {
        let t = tournament();
        for _ in 0..50 {
            let source = Bracket::new(&t, None);
            let rebuilt = Bracket::new_from_binary(&t, &source.binary, None);
            assert_legal(&rebuilt);
            assert_eq!(rebuilt.binary, source.binary);
            assert_eq!(rebuilt.winner_indices(), source.winner_indices());
        }
    }

    #[test]
    fn mutation_preserves_legality() {
        let t = tournament();
        let base = Bracket::new(&t, None);
        for _ in 0..50 {
            assert_legal(&base.mutate(&t, 5.0 / 63.0, None));
        }
    }

    #[test]
    fn a_bracket_scored_against_itself_earns_its_maximum() {
        let t = tournament();
        let b = Bracket::new(&t, None);
        assert!((b.score(&b, None) - b.score).abs() < 1e-9);
    }

    #[test]
    fn expected_value_matches_a_monte_carlo_estimate() {
        // The old expected_value multiplied each pick's points by the
        // probability of winning that game *given the matchup happened*, which
        // overstated it badly (~300 against a true ~228). It should now agree
        // with the mean score against sampled tournaments.
        let t = tournament();
        let bracket = Bracket::new(&t, None);

        let trials = 40_000;
        let table = ScoreTable::default();
        let fast = FastBracket::from_bracket(&bracket);
        let total: f64 = (0..trials)
            .map(|_| {
                let outcome = FastBracket::from_bracket(&Bracket::new(&t, None));
                fast.score_against(&outcome, &table)
            })
            .sum();
        let sampled = total / trials as f64;

        assert!(
            (sampled - bracket.expected_value).abs() < 3.0,
            "sampled {:.2} vs exact {:.2}",
            sampled,
            bracket.expected_value
        );
    }
}
