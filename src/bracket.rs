//This contains all of the simulations using ELO (modified per 538) calculations
//The Game struct is used to simulate a game between two teams and store the results.
//The Bracket struct is used to simulate a whole tournament of 63 games and store the results.
//Both structs also provide a way to create a game from binary data or to extract a binary representation of each.
// the Bracket struct also provides a way to score a bracket against a reference bracket.

use serde::{Serialize}; // Removed Deserialize as it's not used here
use rand::Rng;
use crate::ingest::{Team, TournamentInfo};
use rayon::prelude::*;
use std::cmp; // Added for potential future use with min/max, good practice.

// R0: region_idx * 8 + game_in_round_idx_in_region
// R1: 32 + region_idx * 4 + game_in_round_idx_in_region
// R2: 32 + 16 + region_idx * 2 + game_in_round_idx_in_region
// R3: 32 + 16 + 8 + region_idx * 1 + game_in_round_idx_in_region
// R4: 32 + 16 + 8 + 4 + game_in_round_idx_in_region (Here game_in_round_idx_in_region will be 0 or 1)
// R5: 32 + 16 + 8 + 4 + 2 which is 62.

pub fn get_binary_index_for_game(
    round_idx: usize,
    game_in_round_idx_in_region: usize,
    region_idx: usize,
) -> usize {
    match round_idx {
        0 => region_idx * 8 + game_in_round_idx_in_region, // 0-31
        1 => 32 + region_idx * 4 + game_in_round_idx_in_region, // 32-47
        2 => 32 + 16 + region_idx * 2 + game_in_round_idx_in_region, // 48-55
        3 => 32 + 16 + 8 + region_idx * 1 + game_in_round_idx_in_region, // 56-59
        4 => 32 + 16 + 8 + 4 + game_in_round_idx_in_region, // 60-61; game_in_round_idx_in_region is 0 for E/W, 1 for S/M
        5 => 32 + 16 + 8 + 4 + 2, // 62
        _ => panic!("Invalid round_idx: {}", round_idx),
    }

    pub fn mutate_by_forcing_winner(&self, tournamentinfo: &'a TournamentInfo) -> Bracket<'a> {
        let mut rng = rand::thread_rng();

        // 2. Pick a Random Target Game
        let target_binary_idx = rng.gen_range(0..63);
        let (
            target_round_idx,
            target_game_in_round_idx_in_region,
            target_region_idx,
            _target_region_name, // Usually a string, not used directly beyond get_game_identifier
        ) = get_game_identifier_from_binary_index(target_binary_idx);
        
        // 3. Determine Actual Participants of the Target Game (based on self.binary)
        let (team_a_res, team_b_res) = if target_round_idx == 0 {
            let region_names = ["East", "West", "South", "Midwest"];
            if target_region_idx >= region_names.len() {
                // This case should ideally not be reached if get_game_identifier_from_binary_index is correct
                return Bracket::new(tournamentinfo); // Graceful exit
            }
            let current_region_name = region_names[target_region_idx];
            
            let num_games_per_region_r0 = tournamentinfo.round1.len() / region_names.len(); // typically 8
            let matchup_idx = target_region_idx * num_games_per_region_r0 + target_game_in_round_idx_in_region;

            if matchup_idx >= tournamentinfo.round1.len() {
                 return Bracket::new(tournamentinfo); // Graceful exit
            }
            let matchup_seeds = tournamentinfo.round1[matchup_idx];
            let team1_seed = matchup_seeds[0];
            let team2_seed = matchup_seeds[1];

            (
                find_team_by_seed_and_region_name(tournamentinfo, team1_seed, current_region_name),
                find_team_by_seed_and_region_name(tournamentinfo, team2_seed, current_region_name),
            )
        } else {
            // For rounds > 0, find participants by looking at winners of feeder games from self.binary
            let (p1_res, p2_res) = match target_round_idx {
                1 | 2 | 3 => {
                    let prev_round_idx = target_round_idx - 1;
                    let feeder_game1_idx = target_game_in_round_idx_in_region * 2;
                    let feeder_game2_idx = target_game_in_round_idx_in_region * 2 + 1;
                    (
                        get_game_winner_from_binary(prev_round_idx, feeder_game1_idx, target_region_idx, tournamentinfo, &self.binary, &get_binary_index_for_game),
                        get_game_winner_from_binary(prev_round_idx, feeder_game2_idx, target_region_idx, tournamentinfo, &self.binary, &get_binary_index_for_game),
                    )
                }
                4 => { // Final Four
                    let prev_round_idx = target_round_idx - 1; // R3
                    // target_game_in_round_idx_in_region is 0 for E/W, 1 for S/M
                    // target_region_idx from get_game_identifier for R4 is usually 0.
                    // However, our get_game_winner_from_binary needs specific region_idx for R3 games.
                    // And target_region_idx for R4 in determine_and_set_hilo and find_path is 0 for E/W, 1 for S/M game.
                    let (feeder_r1_idx, feeder_r2_idx) = if target_game_in_round_idx_in_region == 0 { (0,1) } else { (2,3) }; // E/W seeds or S/M seeds
                    (
                        get_game_winner_from_binary(prev_round_idx, 0, feeder_r1_idx, tournamentinfo, &self.binary, &get_binary_index_for_game),
                        get_game_winner_from_binary(prev_round_idx, 0, feeder_r2_idx, tournamentinfo, &self.binary, &get_binary_index_for_game),
                    )
                }
                5 => { // Championship
                    let prev_round_idx = target_round_idx - 1; // R4
                    // Feeders are R4 game 0 (E/W) and R4 game 1 (S/M)
                    // get_game_winner_from_binary for R4 needs game_in_round_idx (0 or 1) and conventional region_idx (0)
                    (
                        get_game_winner_from_binary(prev_round_idx, 0, 0, tournamentinfo, &self.binary, &get_binary_index_for_game),
                        get_game_winner_from_binary(prev_round_idx, 1, 0, tournamentinfo, &self.binary, &get_binary_index_for_game),
                    )
                }
                _ => (Err("Invalid target round".to_string()), Err("Invalid target round".to_string())),
            };
            (p1_res, p2_res)
        };

        let team_a = match team_a_res {
            Ok(t) => t,
            Err(_) => return Bracket::new(tournamentinfo), // Graceful exit
        };
        let team_b = match team_b_res {
            Ok(t) => t,
            Err(_) => return Bracket::new(tournamentinfo), // Graceful exit
        };

        // 4. Select forced_winner
        let forced_winner = if rng.gen_bool(0.5) { team_a } else { team_b };

        // 5. Get Path for forced_winner to Target Game
        // Note: target_region_idx for R4/R5 in find_path_to_game_for_team is defined as:
        // R4: 0 for E/W game, 1 for S/M game. R5: 0 for Champ game.
        // This matches how target_game_in_round_idx_in_region is for R4, and 0 for R5.
        // And target_region_idx from get_game_identifier_from_binary_index for R4 is 0, and for R5 is 0.
        // We need to ensure the correct target_region_idx is passed to find_path.
        // The one from get_game_identifier_from_binary_index is (target_round_idx, gir, path_target_reg_idx, name)
        // where path_target_reg_idx is: R0-3: actual region, R4: 0, R5: 0.
        // The required target_region_idx for find_path_to_game_for_team is:
        // R0-3: actual. R4: 0 if E/W, 1 if S/M. R5: 0.
        // For R4, target_game_in_round_idx_in_region (0 or 1) IS the correct effective region for the path fn.
        let path_target_region_idx_for_find_path = if target_round_idx == 4 {
            target_game_in_round_idx_in_region // 0 for E/W game, 1 for S/M game
        } else {
            target_region_idx // This is correct for R0-3 and R5 (where it's 0)
        };

        let path_to_target_res = find_path_to_game_for_team(
            forced_winner,
            target_round_idx,
            target_game_in_round_idx_in_region,
            path_target_region_idx_for_find_path, 
            tournamentinfo,
        );

        let path_to_target = match path_to_target_res {
            Ok(p) => p,
            Err(_) => return Bracket::new(tournamentinfo), // Graceful exit
        };
        
        // 6. Initialize new_binary
        let mut new_binary = self.binary.clone();

        // 7. Iterate Path and Update new_binary
        for &(r, g_in_reg, path_node_reg_idx) in &path_to_target {
            // The reg_idx for determine_and_set_hilo needs to be the actual region for R0-3,
            // or the conventional one for R4/R5 (0 for E/W, 1 for S/M in R4; 0 for R5).
            // path_node_reg_idx from find_path_to_game_for_team is:
            // R0-3: actual region_idx for that game.
            // R4: 0 if E/W game, 1 if S/M game.
            // R5: 0.
            // This matches the requirements for determine_and_set_hilo_for_forced_winner.
            if determine_and_set_hilo_for_forced_winner(
                r, g_in_reg, path_node_reg_idx, forced_winner, tournamentinfo, &mut new_binary,
                get_binary_index_for_game, get_game_identifier_from_binary_index,
            ).is_err() {
                return Bracket::new(tournamentinfo); // Graceful exit
            }
        }

        // 8. Create and Return New Bracket
        Bracket::new_from_binary(tournamentinfo, new_binary)
    }
}

pub fn get_game_identifier_from_binary_index(
    binary_idx: usize,
) -> (usize, usize, usize, String) {
    let region_names: Vec<&str> = vec!["East", "West", "South", "Midwest"];
    match binary_idx {
        0..=31 => { // Round 1 (R0)
            let region_idx = binary_idx / 8;
            let game_in_round_idx_in_region = binary_idx % 8;
            (0, game_in_round_idx_in_region, region_idx, region_names[region_idx].to_string())
        }
        32..=47 => { // Round 2 (R1)
            let relative_idx = binary_idx - 32;
            let region_idx = relative_idx / 4;
            let game_in_round_idx_in_region = relative_idx % 4;
            (1, game_in_round_idx_in_region, region_idx, region_names[region_idx].to_string())
        }
        48..=55 => { // Round 3 (R2)
            let relative_idx = binary_idx - 48;
            let region_idx = relative_idx / 2;
            let game_in_round_idx_in_region = relative_idx % 2;
            (2, game_in_round_idx_in_region, region_idx, region_names[region_idx].to_string())
        }
        56..=59 => { // Round 4 (R3)
            let relative_idx = binary_idx - 56;
            let region_idx = relative_idx / 1; // or just relative_idx
            let game_in_round_idx_in_region = 0; // Only one game per region in this round
            (3, game_in_round_idx_in_region, region_idx, region_names[region_idx].to_string())
        }
        60..=61 => { // Round 5 (R4) - Final Four
            let game_in_round_idx_in_region = binary_idx - 60; // 0 for E/W, 1 for S/M
            // region_idx is not strictly applicable in the same way,
            // but we can use game_in_round_idx_in_region to determine the grouping.
            // For simplicity, we'll return region_idx 0 and a descriptive name.
            let region_name = if game_in_round_idx_in_region == 0 {
                "FinalFour (E/W)".to_string()
            } else {
                "FinalFour (S/M)".to_string()
            };
            (4, game_in_round_idx_in_region, 0, region_name)
        }
        62 => { // Round 6 (R5) - Championship
            (5, 0, 0, "Championship".to_string())
        }
        _ => panic!("Invalid binary_idx: {}", binary_idx),
    }
}

// Helper to find a team by seed and region name
fn find_team_by_seed_and_region_name<'a>(
    tournament_info: &'a TournamentInfo,
    seed: i32,
    region_name_str: &str,
) -> Result<&'a Team, String> {
    tournament_info
        .teams
        .iter()
        .find(|t| t.seed == seed && t.region == region_name_str)
        .ok_or_else(|| format!("Team not found with seed {} in region {}", seed, region_name_str))
}

// Helper to determine the winner of a game based on binary_representation
pub(crate) fn get_game_winner_from_binary<'a>(
    round_idx: usize,
    game_in_round_idx_in_region: usize, // For R0-R3, this is index within region. For R4, 0=E/W, 1=S/M. For R5, 0=Championship.
    region_idx: usize, // For R0-R3, this is 0-3. For R4/R5, can be a convention (e.g., 0).
    tournament_info: &'a TournamentInfo,
    binary_representation: &Vec<bool>,
    get_binary_idx_fn: &impl Fn(usize, usize, usize) -> usize,
) -> Result<&'a Team, String> {
    let region_names = ["East", "West", "South", "Midwest"]; // Static array

    if round_idx == 0 {
        // Base Case: Round 1
        if region_idx >= region_names.len() {
            return Err(format!("Invalid region_idx {} for R0", region_idx));
        }
        let current_region_name = region_names[region_idx];
        
        // Find the matchup from tournament_info.round1
        // tournament_info.round1 is Vec<[u8; 2]>, storing seed pairs for one region.
        // We need to map game_in_round_idx_in_region to the correct entry.
        // Assuming tournament_info.round1 is structured for a single region and needs to be indexed.
        // This part needs clarification on how tournament_info.round1 is structured for ALL regions.
        // For now, assume round1_matchups are directly indexable by game_in_round_idx_in_region for that region.
        // Let's assume tournament_info.round1 contains seed pairs for *all* R1 games, ordered by region, then game.
        // So, the index into tournament_info.round1 would be region_idx * 8 + game_in_round_idx_in_region
        let matchup_idx_in_tournament_info = region_idx * (tournament_info.round1.len() / region_names.len()) + game_in_round_idx_in_region;

        if matchup_idx_in_tournament_info >= tournament_info.round1.len() {
             return Err(format!(
                "Calculated matchup_idx_in_tournament_info {} out of bounds for R0. region_idx: {}, game_in_round_idx_in_region: {}",
                matchup_idx_in_tournament_info, region_idx, game_in_round_idx_in_region
            ));
        }

        let matchup_seeds = tournament_info.round1[matchup_idx_in_tournament_info];
        let team1_seed = matchup_seeds[0] as i32;
        let team2_seed = matchup_seeds[1] as i32;

        let team1 = find_team_by_seed_and_region_name(tournament_info, team1_seed, current_region_name)?;
        let team2 = find_team_by_seed_and_region_name(tournament_info, team2_seed, current_region_name)?;

        let binary_idx = get_binary_idx_fn(round_idx, game_in_round_idx_in_region, region_idx);
        if binary_idx >= binary_representation.len() {
            return Err(format!("Binary index {} out of bounds.", binary_idx));
        }
        let hilo = binary_representation[binary_idx];

        // Determine winner based on hilo (logic from Game::new_from_binary)
        // In R0, teams are always from the same region.
        let (low_seed_team, high_seed_team) = if team1.seed < team2.seed { (team1, team2) } else { (team2, team1) };
        Ok(if hilo { low_seed_team } else { high_seed_team })
    } else {
        // Recursive Step: Rounds > 0
        let (participant1, participant2) = match round_idx {
            1 | 2 | 3 => { // R1, R2, R3 (regional games leading to regional finals)
                let prev_round_idx = round_idx - 1;
                let feeder_game1_idx_in_region = game_in_round_idx_in_region * 2;
                let feeder_game2_idx_in_region = game_in_round_idx_in_region * 2 + 1;
                
                let p1 = get_game_winner_from_binary(prev_round_idx, feeder_game1_idx_in_region, region_idx, tournament_info, binary_representation, get_binary_idx_fn)?;
                let p2 = get_game_winner_from_binary(prev_round_idx, feeder_game2_idx_in_region, region_idx, tournament_info, binary_representation, get_binary_idx_fn)?;
                (p1, p2)
            }
            4 => { // R4 (Final Four)
                let prev_round_idx = round_idx - 1; // R3
                // game_in_round_idx_in_region: 0 for E/W, 1 for S/M
                // region_idx is not directly used here, but implies which pair of regions
                let (feeder_region1_idx, feeder_region2_idx) = if game_in_round_idx_in_region == 0 { (0, 1) } else { (2, 3) }; // E/W or S/M

                // In R3, there's only one game per region (game_in_round_idx_in_region = 0)
                let p1 = get_game_winner_from_binary(prev_round_idx, 0, feeder_region1_idx, tournament_info, binary_representation, get_binary_idx_fn)?;
                let p2 = get_game_winner_from_binary(prev_round_idx, 0, feeder_region2_idx, tournament_info, binary_representation, get_binary_idx_fn)?;
                (p1, p2)
            }
            5 => { // R5 (Championship)
                let prev_round_idx = round_idx - 1; // R4
                // game_in_round_idx_in_region is 0
                // Feeders are the two R4 games. R4 game_in_round_idx_in_region are 0 and 1.
                // Pass a conventional region_idx (e.g. 0) for R4 games as they are no longer tied to a single initial region.
                let p1 = get_game_winner_from_binary(prev_round_idx, 0, 0, tournament_info, binary_representation, get_binary_idx_fn)?; // E/W winner
                let p2 = get_game_winner_from_binary(prev_round_idx, 1, 0, tournament_info, binary_representation, get_binary_idx_fn)?; // S/M winner
                (p1, p2)
            }
            _ => return Err(format!("Invalid round_idx for recursion: {}", round_idx)),
        };

        let current_game_binary_idx = get_binary_idx_fn(round_idx, game_in_round_idx_in_region, region_idx);
         if current_game_binary_idx >= binary_representation.len() {
            return Err(format!("Binary index {} out of bounds for current game.", current_game_binary_idx));
        }
        let hilo = binary_representation[current_game_binary_idx];

        // Determine winner based on hilo, participant1, participant2
        if participant1.region == participant2.region {
            let (low_seed_team, high_seed_team) = if participant1.seed < participant2.seed { (participant1, participant2) } else { (participant2, participant1) };
            Ok(if hilo { low_seed_team } else { high_seed_team })
        } else { // Different regions (Final Four or Championship)
            let (low_alpha_team, high_alpha_team) = if participant1.region < participant2.region { (participant1, participant2) } else { (participant2, participant1) };
            Ok(if hilo { low_alpha_team } else { high_alpha_team })
        }
    }
}

#[allow(dead_code)] // Potentially unused if only called by the main function below
pub fn determine_and_set_hilo_for_forced_winner<'a>(
    target_round_idx: usize,
    target_game_in_round_idx_in_region: usize,
    target_region_idx: usize,
    forced_winner: &'a Team,
    tournament_info: &'a TournamentInfo,
    binary_representation: &mut Vec<bool>,
    get_binary_idx_fn: impl Fn(usize, usize, usize) -> usize,
    _get_game_identifier_fn: impl Fn(usize) -> (usize, usize, usize, String), // Marked as unused
) -> Result<(), String> {
    let region_names = ["East", "West", "South", "Midwest"];

    let (participant1, participant2) = if target_round_idx == 0 {
        // Base Case: Round 1 - Participants are directly known from tournament_info
        if target_region_idx >= region_names.len() {
            return Err(format!("Invalid target_region_idx {} for R0", target_region_idx));
        }
        let current_region_name = region_names[target_region_idx];
        
        let num_games_per_region_r0 = tournament_info.round1.len() / region_names.len();
        let matchup_idx_in_tournament_info = target_region_idx * num_games_per_region_r0 + target_game_in_round_idx_in_region;

        if matchup_idx_in_tournament_info >= tournament_info.round1.len() {
            return Err(format!(
                "Calculated matchup_idx_in_tournament_info {} out of bounds for R0. target_region_idx: {}, target_game_in_round_idx_in_region: {}",
                matchup_idx_in_tournament_info, target_region_idx, target_game_in_round_idx_in_region
            ));
        }

        let matchup_seeds = tournament_info.round1[matchup_idx_in_tournament_info];
        let team1_seed = matchup_seeds[0] as i32;
        let team2_seed = matchup_seeds[1] as i32;

        let p1 = find_team_by_seed_and_region_name(tournament_info, team1_seed, current_region_name)?;
        let p2 = find_team_by_seed_and_region_name(tournament_info, team2_seed, current_region_name)?;
        (p1, p2)
    } else {
        // Subsequent Rounds: Participants are winners of previous round's games
        // We need to find the winners of the two feeder games.
        match target_round_idx {
            1 | 2 | 3 => { // R1, R2, R3 (regional games)
                let prev_round_idx = target_round_idx - 1;
                let feeder_game1_idx_in_region = target_game_in_round_idx_in_region * 2;
                let feeder_game2_idx_in_region = target_game_in_round_idx_in_region * 2 + 1;
                
                // Pass the original get_binary_idx_fn closure by reference
                let p1 = get_game_winner_from_binary(prev_round_idx, feeder_game1_idx_in_region, target_region_idx, tournament_info, binary_representation, &get_binary_idx_fn)?;
                let p2 = get_game_winner_from_binary(prev_round_idx, feeder_game2_idx_in_region, target_region_idx, tournament_info, binary_representation, &get_binary_idx_fn)?;
                (p1, p2)
            }
            4 => { // R4 (Final Four)
                let prev_round_idx = target_round_idx - 1; // R3
                let (feeder_region1_idx, feeder_region2_idx) = if target_game_in_round_idx_in_region == 0 { (0, 1) } else { (2, 3) }; // E/W or S/M

                let p1 = get_game_winner_from_binary(prev_round_idx, 0, feeder_region1_idx, tournament_info, binary_representation, &get_binary_idx_fn)?;
                let p2 = get_game_winner_from_binary(prev_round_idx, 0, feeder_region2_idx, tournament_info, binary_representation, &get_binary_idx_fn)?;
                (p1, p2)
            }
            5 => { // R5 (Championship)
                let prev_round_idx = target_round_idx - 1; // R4
                // Feeders are the two R4 games (game_idx 0 and 1, conventional region_idx 0)
                let p1 = get_game_winner_from_binary(prev_round_idx, 0, 0, tournament_info, binary_representation, &get_binary_idx_fn)?;
                let p2 = get_game_winner_from_binary(prev_round_idx, 1, 0, tournament_info, binary_representation, &get_binary_idx_fn)?;
                (p1, p2)
            }
            _ => return Err(format!("Invalid target_round_idx: {}", target_round_idx)),
        }
    };

    // Validate forced_winner is one of the participants
    let other_participant = if forced_winner.name == participant1.name {
        participant2
    } else if forced_winner.name == participant2.name {
        participant1
    } else {
        return Err(format!(
            "Forced winner {} (Name: {}) is not one of the determined participants: {} (Name: {}) vs {} (Name: {}) for game at R{}, G{}, Reg{}",
            forced_winner.name, forced_winner.name,
            participant1.name, participant1.name,
            participant2.name, participant2.name,
            target_round_idx, target_game_in_round_idx_in_region, target_region_idx
        ));
    };

    // Determine the required hilo value
    let required_hilo = if forced_winner.region == other_participant.region {
        forced_winner.seed < other_participant.seed
    } else {
        forced_winner.region < other_participant.region // Lexicographical comparison for different regions
    };

    // Update binary_representation
    let binary_idx_to_set = get_binary_idx_fn(target_round_idx, target_game_in_round_idx_in_region, target_region_idx);
    
    if binary_idx_to_set >= binary_representation.len() {
        return Err(format!("Calculated binary_idx_to_set {} is out of bounds for binary_representation (len {}).", binary_idx_to_set, binary_representation.len()));
    }
    binary_representation[binary_idx_to_set] = required_hilo;

    Ok(())
}

fn get_region_idx_from_name(region_name: &str) -> Result<usize, String> {
    match region_name {
        "East" => Ok(0),
        "West" => Ok(1),
        "South" => Ok(2),
        "Midwest" => Ok(3),
        _ => Err(format!("Invalid region name: {}", region_name)),
    }
}

pub(crate) fn find_path_to_game_for_team<'a>(
    forced_winner: &'a Team,
    target_round_idx: usize,
    target_game_in_round_idx_in_region: usize,
    target_region_idx: usize, // R0-3: actual region_idx. R4: 0 for E/W game, 1 for S/M game. R5: 0 for Champ game.
    _tournament_info: &'a TournamentInfo,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let mut path: Vec<(usize, usize, usize)> = Vec::new();

    // 1. Find R0 game for forced_winner
    let r0_region_idx = get_region_idx_from_name(&forced_winner.region)?;
    let mut r0_game_idx_in_region: Option<usize> = None;

    for (idx, seed_matchup) in _tournament_info.round1.iter().enumerate() {
        if forced_winner.seed == seed_matchup[0] || forced_winner.seed == seed_matchup[1] {
            r0_game_idx_in_region = Some(idx);
            break;
        }
    }

    let r0_game_idx_in_region = r0_game_idx_in_region.ok_or_else(|| {
        format!(
            "Team {} (Seed: {}, Region: {}) not found in any R0 matchup.",
            forced_winner.name, forced_winner.seed, forced_winner.region
        )
    })?;

    path.push((0, r0_game_idx_in_region, r0_region_idx));

    // 2. Forward Iteration to Build Path
    while let Some(&(current_round, current_game_in_reg, current_reg_idx_on_path)) = path.last() {
        if current_round == target_round_idx && 
           current_game_in_reg == target_game_in_round_idx_in_region && 
           current_reg_idx_on_path == target_region_idx {
            break; // Reached the target game
        }

        if current_round >= target_round_idx {
            // Overshot or reached target_round but not the exact game.
            // This case should be caught by the final validation if target not met.
            break;
        }

        let next_round = current_round + 1;
        let (next_game_in_reg, next_reg_idx_for_path): (usize, usize) = match next_round {
            1 | 2 | 3 => { // Intra-regional: R1, R2, R3
                (current_game_in_reg / 2, current_reg_idx_on_path)
            }
            4 => { // R4 (Final Four)
                // current_reg_idx_on_path is the team's original region (0-3)
                let ff_game_slot = match current_reg_idx_on_path {
                    0 | 1 => 0, // East or West team goes to Final Four game slot 0
                    2 | 3 => 1, // South or Midwest team goes to Final Four game slot 1
                    _ => return Err(format!("Invalid region index {} for R3 winner feeding into R4", current_reg_idx_on_path)),
                };
                // For path purposes, the "region index" of an R4 game is its game slot (0 or 1)
                (ff_game_slot, ff_game_slot) 
            }
            5 => { // R5 (Championship)
                // current_game_in_reg for R4 is 0 or 1. Both feed into R5 game 0.
                // current_reg_idx_on_path for R4 is 0 or 1.
                (0, 0) // Championship game is (0,0) in its own context
            }
            _ => return Err(format!("Path construction attempted to go beyond R5 to round {}", next_round)),
        };
        path.push((next_round, next_game_in_reg, next_reg_idx_for_path));
    }

    // 3. Validation
    if let Some(&(final_r, final_g, final_reg)) = path.last() {
        if final_r == target_round_idx &&
           final_g == target_game_in_round_idx_in_region &&
           final_reg == target_region_idx {
            Ok(path)
        } else {
            Err(format!(
                "Could not find a valid path for {} to target game (R{}, G{}, RegIdx{}). Path ended at (R{}, G{}, RegIdx{}). Team's R0 region: {}",
                forced_winner.name,
                target_round_idx, target_game_in_round_idx_in_region, target_region_idx,
                final_r, final_g, final_reg,
                forced_winner.region
            ))
        }
    } else {
        Err("Path construction failed: path is empty.".to_string()) // Should not happen if R0 game is found
    }
}


#[derive(Serialize, Debug, Clone)]
pub struct Game<'a> {
    pub team1: &'a Team,
    pub team2: &'a Team,
    pub team1prob: f64,
    pub team2prob: f64,
    pub winnerprob: f64,
    pub winner: &'a Team,
    pub hilo: bool, //did the lower seed (or the region first in alphabetical order win) win?
}

impl<'a> Game<'a> {
    pub fn new(team1: &'a Team, team2: &'a Team) -> Game<'a> {
        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let mut rng = rand::thread_rng();
        let rand_num: f64 = rng.gen(); 

        let winner: &'a Team = if rand_num < team1prob {
            team1
        } else {
            team2
        };

        let winnerprob = if rand_num < team1prob {
            team1prob
        } else {
            team2prob
        };

        let hilo: bool = if team1.region == team2.region {
            team1.seed < team2.seed
        } else {
            team1.region < team2.region
        };
        Game {
            team1,
            team2,
            team1prob,
            team2prob,
            winnerprob,
            winner,
            hilo,
        }
    }

    pub fn new_from_binary(team1: &'a Team, team2: &'a Team, hilo: bool) -> Game<'a> {
        let low_seed_team: &'a Team = if team1.seed < team2.seed { team1 } else { team2 };
        let high_seed_team: &'a Team = if team1.seed < team2.seed { team2 } else { team1 };
        let low_alpha_team: &'a Team = if team1.region < team2.region { team1 } else { team2 };
        let high_alpha_team: &'a Team = if team1.region < team2.region { team2 } else { team1 };

        let winner: &'a Team = match hilo {
            true if team1.region == team2.region => low_seed_team,
            false if team1.region == team2.region => high_seed_team,
            true if team1.region != team2.region => low_alpha_team,
            false if team1.region != team2.region => high_alpha_team,
            _ => panic!("Something went wrong in the hilo logic for new_from_binary"),
        };

        let rating_diff = team1.rating as f64 - team2.rating as f64;
        let team1prob: f64 = 1.0 / (1.0 + 10.0f64.powf(-1.0 * rating_diff * 30.464 / 400.0));
        let team2prob: f64 = 1.0 - team1prob;

        let winnerprob = if winner == team1 { team1prob } else { team2prob };

        Game {
            team1,
            team2,
            team1prob,
            team2prob,
            winnerprob,
            winner,
            hilo,
        }
    }

    #[allow(dead_code)] // It's used in main.rs, but cargo check in sandbox might not see it
    pub fn print(&self) {
        println!("{:?}", self);
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct Bracket<'a> {
    pub round1: Vec<Game<'a>>, 
    pub round2: Vec<Game<'a>>, 
    pub round3: Vec<Game<'a>>, 
    pub round4: Vec<Game<'a>>, 
    pub round5: Vec<Game<'a>>, 
    pub round6: Vec<Game<'a>>, 
    pub winner: &'a Team, 
    pub prob: f64,  
    pub score: f64, 
    pub sim_score: f64, 
    pub expected_value: f64, 
    pub binary: Vec<bool>,
}

impl<'a> PartialEq for Bracket<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.binary == other.binary
    }
}

impl<'a> Bracket<'a> {
    pub fn new(tournamentinfo: &'a TournamentInfo) -> Bracket<'a> {
        let mut games1: Vec<Game<'a>> = Vec::with_capacity(32);
        let mut games2: Vec<Game<'a>> = Vec::with_capacity(16);
        let mut games3: Vec<Game<'a>> = Vec::with_capacity(8);
        let mut games4: Vec<Game<'a>> = Vec::with_capacity(4);
        let mut games5: Vec<Game<'a>> = Vec::with_capacity(2);
        let mut games6: Vec<Game<'a>> = Vec::with_capacity(1);

        let mut games1winners: Vec<&'a Team> = Vec::with_capacity(32);
        let mut games2winners: Vec<&'a Team> = Vec::with_capacity(16);
        let mut games3winners: Vec<&'a Team> = Vec::with_capacity(8);
        let mut games4winners: Vec<&'a Team> = Vec::with_capacity(4); 
        let mut games5winners: Vec<&'a Team> = Vec::with_capacity(2); 
        let mut games6winners: Vec<&'a Team> = Vec::with_capacity(1); 

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;
        let mut binary: Vec<bool> = Vec::with_capacity(63);
        let region_names: Vec<&str> = vec!["East", "West", "South", "Midwest"];
        
        for &region_name in &region_names {
            for r1_matchup_seeds in tournamentinfo.round1 { 
                let team1 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[0] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[0]));
                let team2 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[1] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[1]));
                let game = Game::new(team1, team2);
                games1.push(game.clone());
                games1winners.push(game.winner);
                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (1.0 + game.winner.seed as f64);
                binary.push(game.hilo);
            }
        }
        assert!(games1winners.len() == 32);

        let mut r1_winner_idx_offset = 0;
        for _ in 0..region_names.len() { 
            for _ in 0..4 { 
                let team1 = games1winners[r1_winner_idx_offset];
                let team2 = games1winners[r1_winner_idx_offset + 1];
                let game = Game::new(team1, team2);
                games2.push(game.clone());
                games2winners.push(game.winner);
                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (2.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                r1_winner_idx_offset += 2; 
            }
        }
        assert!(games2winners.len() == 16);
        
        let mut r2_winner_idx_offset = 0; 
        for _ in 0..region_names.len() { 
            for _ in 0..2 { 
                let team1 = games2winners[r2_winner_idx_offset];
                let team2 = games2winners[r2_winner_idx_offset + 1];
                let game = Game::new(team1, team2);
                games3.push(game.clone());
                games3winners.push(game.winner);
                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += game.winnerprob * (4.0 + game.winner.seed as f64);
                binary.push(game.hilo);
                r2_winner_idx_offset += 2;
            }
        }
        assert!(games3winners.len() == 8);

        let mut r3_winner_idx_offset = 0; 
        for _ in 0..region_names.len() { 
            let team1 = games3winners[r3_winner_idx_offset];
            let team2 = games3winners[r3_winner_idx_offset + 1];
            let game = Game::new(team1, team2);
            games4.push(game.clone());
            games4winners.push(game.winner); 
            prob *= game.winnerprob;
            score += 8.0 * game.winner.seed as f64;
            expected_value += game.winnerprob * (8.0 * game.winner.seed as f64);
            binary.push(game.hilo);
            r3_winner_idx_offset += 2;
        }
        assert!(games4winners.len() == 4); // E, W, S, M champions

        let team1_ew = games4winners[0]; 
        let team2_ew = games4winners[1]; 
        let game_ew = Game::new(team1_ew, team2_ew);
        games5.push(game_ew.clone());
        games5winners.push(game_ew.winner);
        prob *= game_ew.winnerprob;
        score += 16.0 * game_ew.winner.seed as f64;
        expected_value += game_ew.winnerprob * (16.0 * game_ew.winner.seed as f64);
        binary.push(game_ew.hilo);

        let team1_sm = games4winners[2]; 
        let team2_sm = games4winners[3]; 
        let game_sm = Game::new(team1_sm, team2_sm);
        games5.push(game_sm.clone());
        games5winners.push(game_sm.winner);
        prob *= game_sm.winnerprob;
        score += 16.0 * game_sm.winner.seed as f64;
        expected_value += game_sm.winnerprob * (16.0 * game_sm.winner.seed as f64);
        binary.push(game_sm.hilo);
        assert!(games5winners.len() == 2); 
        
        let game_champ = Game::new(games5winners[0], games5winners[1]);
        games6.push(game_champ.clone());
        games6winners.push(game_champ.winner);
        prob *= game_champ.winnerprob;
        score += 32.0 * game_champ.winner.seed as f64;
        expected_value += game_champ.winnerprob * (32.0 * game_champ.winner.seed as f64);
        binary.push(game_champ.hilo);

        assert!(games6winners.len() == 1);
        assert!(binary.len() == 63);

        Bracket{
            round1: games1, round2: games2, round3: games3, round4: games4, round5: games5, round6: games6,
            winner: games6winners[0], prob, score, sim_score: 0.0, expected_value, binary,
        }
    }

    pub fn score(&self, referencebracket: &Bracket<'a>) -> f64{
        let mut current_score: f64 = 0.0;
        for i in 0..32{ if self.round1[i].winner == referencebracket.round1[i].winner{ current_score += 1.0 + self.round1[i].winner.seed as f64; } }
        for i in 0..16{ if self.round2[i].winner == referencebracket.round2[i].winner{ current_score += 2.0 + self.round2[i].winner.seed as f64; } }
        for i in 0..8{  if self.round3[i].winner == referencebracket.round3[i].winner{ current_score += 4.0 + self.round3[i].winner.seed as f64; } }
        for i in 0..4{  if self.round4[i].winner == referencebracket.round4[i].winner{ current_score += 8.0 * self.round4[i].winner.seed as f64; } }
        for i in 0..2{  if self.round5[i].winner == referencebracket.round5[i].winner{ current_score += 16.0 * self.round5[i].winner.seed as f64;} }
        if self.round6[0].winner == referencebracket.round6[0].winner{ current_score += 32.0 * self.round6[0].winner.seed as f64; }
        current_score
    }

    pub fn new_from_binary(tournamentinfo: &'a TournamentInfo, binary_string: Vec<bool>) -> Bracket<'a>{
        let mut games1: Vec<Game<'a>> = Vec::with_capacity(32);
        let mut games2: Vec<Game<'a>> = Vec::with_capacity(16);
        let mut games3: Vec<Game<'a>> = Vec::with_capacity(8);
        let mut games4: Vec<Game<'a>> = Vec::with_capacity(4);
        let mut games5: Vec<Game<'a>> = Vec::with_capacity(2);
        let mut games6: Vec<Game<'a>> = Vec::with_capacity(1);

        let mut games1winners: Vec<&'a Team> = Vec::with_capacity(32);
        let mut games2winners: Vec<&'a Team> = Vec::with_capacity(16);
        let mut games3winners: Vec<&'a Team> = Vec::with_capacity(8);
        let mut games4winners: Vec<&'a Team> = Vec::with_capacity(4);
        let mut games5winners: Vec<&'a Team> = Vec::with_capacity(2);
        let mut games6winners: Vec<&'a Team> = Vec::with_capacity(1);

        let mut prob: f64 = 1.0;
        let mut score: f64 = 0.0;
        let mut expected_value = 0.0;

        assert!(binary_string.len() == 63, "Binary string must be 63 characters long");
        let mut hilo_iterator = binary_string.iter();
        let region_names: Vec<&str> = vec!["East", "West", "South", "Midwest"]; 
        
        for &region_name in &region_names { 
            for r1_matchup_seeds in tournamentinfo.round1 {
                let team1 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[0] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[0]));
                let team2 = tournamentinfo.teams.iter().find(|t| t.region == region_name && t.seed == r1_matchup_seeds[1] as i32).unwrap_or_else(|| panic!("R1 Team not found: {}, seed {}", region_name, r1_matchup_seeds[1]));
                let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R1"));
                games1.push(game.clone());
                games1winners.push(game.winner);
                prob *= game.winnerprob;
                score += 1.0 + game.winner.seed as f64;
                expected_value += (1.0 + game.winner.seed as f64) * game.winnerprob;
            }
        }
        assert!(games1winners.len() == 32);

        let mut r1_winner_idx_offset = 0;
        for _ in &region_names {
            for _ in 0..4 {
                let team1 = games1winners[r1_winner_idx_offset];
                let team2 = games1winners[r1_winner_idx_offset + 1];
                let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R2"));
                games2.push(game.clone());
                games2winners.push(game.winner);
                prob *= game.winnerprob;
                score += 2.0 + game.winner.seed as f64;
                expected_value += (2.0 + game.winner.seed as f64) * game.winnerprob;
                r1_winner_idx_offset += 2;
            }
        }
        assert!(games2winners.len() == 16);
        
        let mut r2_winner_idx_offset = 0;
        for _ in &region_names {
            for _ in 0..2 {
                let team1 = games2winners[r2_winner_idx_offset];
                let team2 = games2winners[r2_winner_idx_offset + 1];
                let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R3"));
                games3.push(game.clone());
                games3winners.push(game.winner);
                prob *= game.winnerprob;
                score += 4.0 + game.winner.seed as f64;
                expected_value += (4.0 + game.winner.seed as f64) * game.winnerprob;
                r2_winner_idx_offset += 2;
            }
        }
        assert!(games3winners.len() == 8);

        let mut r3_winner_idx_offset = 0;
        for _ in &region_names {
            let team1 = games3winners[r3_winner_idx_offset];
            let team2 = games3winners[r3_winner_idx_offset + 1];
            let game = Game::new_from_binary(team1, team2, *hilo_iterator.next().expect("Binary string exhausted prematurely in R4"));
            games4.push(game.clone());
            games4winners.push(game.winner);
            prob *= game.winnerprob;
            score += 8.0 * game.winner.seed as f64;
            expected_value += (8.0 * game.winner.seed as f64) * game.winnerprob;
            r3_winner_idx_offset += 2;
        }
        assert!(games4winners.len() == 4);

        let team1_ew = games4winners[0]; 
        let team2_ew = games4winners[1]; 
        let game_ew = Game::new_from_binary(team1_ew, team2_ew, *hilo_iterator.next().expect("Binary string exhausted prematurely in F4 EW"));
        games5.push(game_ew.clone());
        games5winners.push(game_ew.winner);
        prob *= game_ew.winnerprob;
        score += 16.0 * game_ew.winner.seed as f64;
        expected_value += (16.0 * game_ew.winner.seed as f64) * game_ew.winnerprob;
        
        let team1_sm = games4winners[2]; 
        let team2_sm = games4winners[3]; 
        let game_sm = Game::new_from_binary(team1_sm, team2_sm, *hilo_iterator.next().expect("Binary string exhausted prematurely in F4 SM"));
        games5.push(game_sm.clone());
        games5winners.push(game_sm.winner);
        prob *= game_sm.winnerprob;
        score += 16.0 * game_sm.winner.seed as f64;
        expected_value += (16.0 * game_sm.winner.seed as f64) * game_sm.winnerprob;
        assert!(games5winners.len() == 2);
        
        let game_champ = Game::new_from_binary(games5winners[0], games5winners[1], *hilo_iterator.next().expect("Binary string exhausted prematurely in Championship"));
        games6.push(game_champ.clone());
        games6winners.push(game_champ.winner);
        prob *= game_champ.winnerprob;
        score += 32.0 * game_champ.winner.seed as f64;
        expected_value += (32.0 * game_champ.winner.seed as f64) * game_champ.winnerprob;
        assert!(games6winners.len() == 1);
        assert!(hilo_iterator.next().is_none(), "Hilo iterator was not fully consumed.");

        Bracket{
            round1: games1, round2: games2, round3: games3, round4: games4, round5: games5, round6: games6,
            winner: games6winners[0], prob, score, sim_score: 0.0, expected_value,
            binary: binary_string, 
        }
    }

    // Updated mutate to use the new forcing winner strategy
    pub fn mutate(&self, tournamentinfo: &'a TournamentInfo) -> Bracket<'a> {
        self.mutate_by_forcing_winner(tournamentinfo)
    }

    // Updated create_n_children to use the new mutate signature
    pub fn create_n_children(&self, tournamentinfo: &'a TournamentInfo, n: usize) -> Vec<Bracket<'a>> {
        (0..n)
            .into_par_iter()
            .map(|_| self.mutate(tournamentinfo))
            .collect()
    }

    #[allow(dead_code)]
    pub fn hamming_distance(&self, other: &Bracket<'a>) -> usize{
        self.binary.iter().zip(other.binary.iter()).filter(|&(a,b)| a != b).count()
    }
    
    pub fn pretty_print(&self){
        println!("Round 1");
        for game in &self.round1{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nRound 2");
        for game in &self.round2{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nSweet 16");
        for game in &self.round3{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nElite 8");
        for game in &self.round4{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nFinal Four");
        for game in &self.round5{ println!("{} {}", game.winner.seed, game.winner.name); }
        println!("\nChampionship");
        println!("{} {} wins!", self.round6[0].winner.seed, self.round6[0].winner.name);
        println!("Expected Value: {}", self.expected_value);
        println!("Maximum Score: {}", self.score);
        println!();
    }
}

#[allow(dead_code)]
pub fn random63bool() -> Vec<bool>{
    let mut rng = rand::thread_rng();
    (0..63).map(|_| rng.gen::<bool>()).collect()
}

#[cfg(test)]
mod tests {
    use super::*; // Access functions and structs from the parent module
    use crate::ingest::TournamentInfo; // Explicitly import if needed, but super::* should cover it.

    // Helper function to load tournament data
    fn setup_tournament_info() -> TournamentInfo {
        // Assuming the CSV file is at the root of the crate.
        // Adjust path if necessary for the test environment.
        TournamentInfo::initialize("fivethirtyeight_ncaa_forecasts.csv")
    }

    // Helper to find a specific team for testing
    fn find_test_team<'a>(ti: &'a TournamentInfo, name: &str) -> &'a Team {
        ti.teams.iter().find(|t| t.name == name).expect(&format!("Test team '{}' not found", name))
    }
    
    // Helper to find a specific team by seed and region for testing
    fn find_test_team_by_seed_region<'a>(ti: &'a TournamentInfo, seed: i32, region: &str) -> &'a Team {
        ti.teams.iter().find(|t| t.seed == seed && t.region == region)
            .expect(&format!("Test team with seed {} in region '{}' not found", seed, region))
    }


    #[test]
    fn test_binary_index_round_trip() {
        // Test case 1: R0G0 East (Region 0, Game 0)
        let r0g0e_idx = get_binary_index_for_game(0, 0, 0);
        assert_eq!(r0g0e_idx, 0);
        let (r, g, reg, _) = get_game_identifier_from_binary_index(r0g0e_idx);
        assert_eq!((r, g, reg), (0, 0, 0));

        // Test case 2: R1G3 Midwest (Region 3, Game 3 in round) -> 32 + 3*4 + 3 = 32 + 12 + 3 = 47
        let r1g3m_idx = get_binary_index_for_game(1, 3, 3);
        assert_eq!(r1g3m_idx, 47);
        let (r, g, reg, _) = get_game_identifier_from_binary_index(r1g3m_idx);
        assert_eq!((r, g, reg), (1, 3, 3));

        // Test case 3: R3G0 West (Region 1, Game 0 in round) -> 32+16+8 + 1*1 + 0 = 56 + 1 = 57
        let r3g0w_idx = get_binary_index_for_game(3, 0, 1);
        assert_eq!(r3g0w_idx, 57);
        let (r, g, reg, _) = get_game_identifier_from_binary_index(r3g0w_idx);
        assert_eq!((r, g, reg), (3, 0, 1));
        
        // Test case 4: R4G0 (Final Four E/W game, game_in_round_idx_in_region=0, conventional region_idx=0) -> 32+16+8+4 + 0 = 60
        let r4g0_idx = get_binary_index_for_game(4, 0, 0); // Here region_idx is conventional for FF games
        assert_eq!(r4g0_idx, 60);
        let (r, g, reg, name) = get_game_identifier_from_binary_index(r4g0_idx);
        assert_eq!((r, g, reg), (4, 0, 0)); // reg is conventional
        assert!(name.contains("FinalFour"));

        // Test case 5: R5G0 (Championship) -> 32+16+8+4+2 = 62
        let r5g0_idx = get_binary_index_for_game(5, 0, 0);
        assert_eq!(r5g0_idx, 62);
        let (r, g, reg, name) = get_game_identifier_from_binary_index(r5g0_idx);
        assert_eq!((r, g, reg), (5, 0, 0));
        assert!(name.contains("Championship"));
    }

    #[test]
    fn test_find_path_east_seed1_to_r3_east() {
        let ti = setup_tournament_info();
        let team_alabama = find_test_team_by_seed_region(&ti, 1, "South"); // Alabama was overall #1, in South
        // Let's use a known #1 seed from East for clarity, e.g. Purdue from Midwest in 2023, or find actual East #1
        // For this test, we need an East #1. If not present, this test might be fragile.
        // Assuming an East #1 seed exists and is named "Purdue" for example, or find dynamically.
        // Let's find the actual #1 seed in East from the data.
        let east_seed1 = ti.teams.iter().find(|t| t.region == "East" && t.seed == 1)
            .expect("East #1 seed not found for test_find_path_east_seed1_to_r3_east");

        // Target: R3G0 East (Elite Eight game for East champion)
        // R3 = round_idx 3, G0 = game_in_round_idx_in_region 0, East = region_idx 0
        let path_res = find_path_to_game_for_team(east_seed1, 3, 0, 0, &ti);
        assert!(path_res.is_ok(), "Path finding failed: {:?}", path_res.err());
        let path = path_res.unwrap();

        assert_eq!(path.len(), 4, "Path length should be 4 (R0, R1, R2, R3)");
        // Path: (R0, G0, Reg0), (R1, G0, Reg0), (R2, G0, Reg0), (R3, G0, Reg0)
        // Seed 1 plays in game 0 of R0 (1v16)
        // Winner of R0G0 plays in game 0 of R1 (1/16 vs 8/9 winner)
        // Winner of R1G0 plays in game 0 of R2 (Sweet 16)
        // Winner of R2G0 plays in game 0 of R3 (Elite 8)
        let expected_path = vec![
            (0, 0, 0), // R0, G0 (1v16), East
            (1, 0, 0), // R1, G0 (winner of 1v16 vs 8v9), East
            (2, 0, 0), // R2, G0 (Sweet 16 game), East
            (3, 0, 0), // R3, G0 (Elite 8 game), East
        ];
        assert_eq!(path, expected_path);
    }

    #[test]
    fn test_find_path_west_seed6_to_r5_championship() {
        let ti = setup_tournament_info();
        let west_seed6 = ti.teams.iter().find(|t| t.region == "West" && t.seed == 6)
            .expect("West #6 seed not found for test.");
        
        // Target: R5G0 Championship (Round 5, Game 0, Conventional Region 0)
        let path_res = find_path_to_game_for_team(west_seed6, 5, 0, 0, &ti);
        assert!(path_res.is_ok(), "Path finding failed: {:?}", path_res.err());
        let path = path_res.unwrap();

        assert_eq!(path.len(), 6, "Path length should be 6 (R0-R5)");
        // R0: (0, game_idx_for_seed6, region_idx_West=1)
        // Seed 6 plays in game index 5 (6v11 matchup) in its R0 region.
        let r0_game_idx_west_seed6 = _tournament_info.round1.iter().position(|&m| m[0] == 6 || m[1] == 6).unwrap();

        // Path: (R0, G_seed6, Reg_West), (R1, G_path, Reg_West), (R2, G_path, Reg_West), (R3, G_path, Reg_West), 
        //       (R4, G_E/W, Reg_E/W_slot), (R5, G0, Reg_Champ)
        // West region is idx 1. Path for R4 should be game slot 0 (E/W).
        let expected_path_nodes = vec![
            (0, r0_game_idx_west_seed6, 1), // R0 West (idx 1), game for seed 6 (idx 5)
            (1, r0_game_idx_west_seed6 / 2, 1), // R1 West
            (2, (r0_game_idx_west_seed6 / 2) / 2, 1), // R2 West
            (3, ((r0_game_idx_west_seed6 / 2) / 2) / 2, 1), // R3 West (Regional Champ)
            (4, 0, 0), // R4 Final Four (E/W game, slot 0, conventional region_idx 0)
            (5, 0, 0), // R5 Championship (game 0, conventional region_idx 0)
        ];
        assert_eq!(path, expected_path_nodes);
    }

    #[test]
    fn test_find_path_midwest_to_ew_final_four_error() {
        let ti = setup_tournament_info();
        let midwest_team = ti.teams.iter().find(|t| t.region == "Midwest" && t.seed == 1)
            .expect("Midwest #1 seed not found for test.");

        // Target: R4G0 (E/W Final Four game, target_region_idx for path fn = 0)
        let path_res = find_path_to_game_for_team(midwest_team, 4, 0, 0, &ti);
        assert!(path_res.is_err(), "Path should be invalid for Midwest team to E/W FF game.");
        if let Err(e) = path_res {
            println!("Midwest to E/W FF error: {}", e); // Optional: print error for confirmation
            assert!(e.contains("Could not find a valid path")); // Check for specific error message if desired
        }
    }
    
    #[test]
    fn test_mutation_logic_propagation() {
        let ti = setup_tournament_info();

        // a. Initial binary (all false means higher seed/alphabetical region wins)
        let initial_binary = vec![false; 63]; 
        
        // b. Select target_game: R2G0 East (Sweet 16, first game in East)
        // R2 = round 2, G0 = game 0 in region, East = region_idx 0
        let target_r = 2;
        let target_g_in_reg = 0;
        let target_reg = 0; // East

        // c. Determine expected participants of R2G0 East based on initial_binary (all false)
        // R2G0 East is fed by R1G0 East and R1G1 East
        // R1G0 East is fed by R0G0 East and R0G1 East
        // R1G1 East is fed by R0G2 East and R0G3 East
        // With all false, higher seeds win.
        // R0G0 East (1v16) -> 1 wins. R0G1 East (8v9) -> 8 wins. R1G0 East (1v8) -> 1 wins.
        // R0G2 East (5v12) -> 5 wins. R0G3 East (4v13) -> 4 wins. R1G1 East (5v4) -> 4 wins.
        // So, R2G0 East would be Seed 1 (East) vs Seed 4 (East)
        let p1_r1g0 = get_game_winner_from_binary(1, 0, 0, &ti, &initial_binary, &get_binary_index_for_game).unwrap();
        let p2_r1g1 = get_game_winner_from_binary(1, 1, 0, &ti, &initial_binary, &get_binary_index_for_game).unwrap();
        
        // d. Select forced_winner: Let's force p1_r1g0 (expected Seed 1 of East) to win up to R2G0 East.
        let forced_winner = p1_r1g0; 
        assert_eq!(forced_winner.seed, 1); // Sanity check based on all-false binary
        assert_eq!(forced_winner.region, "East");

        // e. Get path for forced_winner to target_game
        let path = find_path_to_game_for_team(forced_winner, target_r, target_g_in_reg, target_reg, &ti).unwrap();
        let expected_path_r2g0_east_s1 = vec![(0,0,0), (1,0,0), (2,0,0)];
        assert_eq!(path, expected_path_r2g0_east_s1);

        // f. Clone initial binary
        let mut mutated_binary = initial_binary.clone();

        // g. Iterate path and set hilo using determine_and_set_hilo_for_forced_winner
        for &(r, g, reg_idx_on_path) in &path {
            determine_and_set_hilo_for_forced_winner(
                r, g, reg_idx_on_path, forced_winner, &ti, &mut mutated_binary,
                get_binary_index_for_game, get_game_identifier_from_binary_index
            ).expect("determine_and_set_hilo_for_forced_winner failed during test mutation setup");
        }

        // h. Create final_bracket (not strictly necessary if we check mutated_binary directly, but good for full picture)
        // let final_bracket = Bracket::new_from_binary(&ti, mutated_binary.clone());

        // i. Assertions
        // i.1. Verify forced_winner is the winner of the target_game using mutated_binary
        let winner_of_target_game = get_game_winner_from_binary(
            target_r, target_g_in_reg, target_reg, &ti, &mutated_binary, &get_binary_index_for_game
        ).unwrap();
        assert_eq!(winner_of_target_game.name, forced_winner.name, "Forced winner should win the target game.");

        // i.2. For every game on the path, verify forced_winner is the winner
        for &(r, g, reg_idx_on_path) in &path {
            let game_winner_on_path = get_game_winner_from_binary(
                r, g, reg_idx_on_path, &ti, &mutated_binary, &get_binary_index_for_game
            ).unwrap();
            assert_eq!(game_winner_on_path.name, forced_winner.name, "Forced winner should win game (R{}, G{}, Reg{}) on path.", r, g, reg_idx_on_path);
        
            // i.3. Verify hilo bits in mutated_binary for path games
            // This requires finding the *other* participant of game (r,g,reg_idx_on_path) assuming forced_winner won games leading to it
            let (p1, p2) = if r == 0 {
                let region_names = ["East", "West", "South", "Midwest"];
                let current_region_name = region_names[reg_idx_on_path];
                let matchup_idx = reg_idx_on_path * (ti.round1.len() / 4) + g;
                let matchup_seeds = ti.round1[matchup_idx];
                (find_team_by_seed_and_region_name(&ti, matchup_seeds[0], current_region_name).unwrap(),
                 find_team_by_seed_and_region_name(&ti, matchup_seeds[1], current_region_name).unwrap())
            } else {
                 // Determine feeder games based on r, g, reg_idx_on_path
                let (feeder1_r, feeder1_g, feeder1_reg) = Bracket::get_feeder_game_info(r, g, reg_idx_on_path, 0);
                let (feeder2_r, feeder2_g, feeder2_reg) = Bracket::get_feeder_game_info(r, g, reg_idx_on_path, 1);

                (get_game_winner_from_binary(feeder1_r, feeder1_g, feeder1_reg, &ti, &mutated_binary, &get_binary_index_for_game).unwrap(),
                 get_game_winner_from_binary(feeder2_r, feeder2_g, feeder2_reg, &ti, &mutated_binary, &get_binary_index_for_game).unwrap())
            };
            
            let other_participant = if forced_winner.name == p1.name { p2 } else { p1 };
            let expected_hilo = if forced_winner.region == other_participant.region {
                forced_winner.seed < other_participant.seed
            } else {
                forced_winner.region < other_participant.region
            };
            let game_binary_idx = get_binary_index_for_game(r, g, reg_idx_on_path);
            assert_eq!(mutated_binary[game_binary_idx], expected_hilo, "Hilo bit incorrect for game (R{}, G{}, Reg{}) on path.", r,g,reg_idx_on_path);
        }
    }

    // Dummy get_feeder_game_info for the test - needs to be part of Bracket or a test utility
    // This is a simplified version and might need actual logic from Bracket struct if it exists,
    // or be built out based on the logic in get_game_winner_from_binary.
    impl<'a> Bracket<'a> {
        fn get_feeder_game_info(round_idx: usize, game_in_round_idx_in_region: usize, region_idx: usize, feeder_num: usize) -> (usize, usize, usize) {
            let prev_round_idx = round_idx -1;
            match round_idx {
                1 | 2 | 3 => {
                    let feeder_game = game_in_round_idx_in_region * 2 + feeder_num;
                    (prev_round_idx, feeder_game, region_idx)
                }
                4 => { // Final Four
                    let (feeder_reg1, feeder_reg2) = if game_in_round_idx_in_region == 0 {(0,1)} else {(2,3)};
                    if feeder_num == 0 { (prev_round_idx, 0, feeder_reg1) } else { (prev_round_idx, 0, feeder_reg2)}
                }
                5 => { // Championship
                     if feeder_num == 0 { (prev_round_idx, 0, 0) } else { (prev_round_idx, 1, 0)} // R4G0 and R4G1
                }
                _ => panic!("Cannot get feeders for R0"),
            }
        }
    }
}
