//! Game results from BartTorvik.
//!
//! The season's game log is one CSV request — `getgamestats.php?year=YYYY&csv=1`
//! returns every Division I game with date, opponent, site, and final score.
//! That matters more than convenience: the day-by-day scoreboard sources fetch
//! 158 URLs per season, and any one of them failing takes the season with it.
//!
//! The feed has no header row and no key. Each game appears **twice**, once from
//! each team's point of view, distinguished by the site column (`H`/`A`/`N`).
//! Home-perspective rows carry the whole game, so the away rows are dropped;
//! neutral-site games have two `N` rows and are deduplicated by matchup.

use crate::game_result::GameResult;
use crate::names;
use chrono::NaiveDate;
use std::collections::HashSet;

/// Column positions in the `getgamestats.php` CSV. The feed is headerless and
/// has carried 31 columns for years; naming the ones we read keeps the parser
/// readable and makes a layout change fail loudly rather than silently.
const COL_DATE: usize = 0;
const COL_TEAM: usize = 2;
const COL_CONF: usize = 3;
const COL_OPPONENT: usize = 4;
const COL_SITE: usize = 5;
const COL_RESULT: usize = 6;
const COL_OPP_CONF: usize = 20;
const MIN_COLUMNS: usize = 21;

/// `year` is the season's *ending* calendar year: 2027 for the 2026-27 season.
pub fn season_url(year: i32) -> String {
    format!("https://barttorvik.com/getgamestats.php?year={}&csv=1", year)
}

/// One row's view of a game, before the two views are reconciled.
struct Row<'a> {
    date: NaiveDate,
    team: &'a str,
    team_conf: &'a str,
    opponent: &'a str,
    opponent_conf: &'a str,
    site: Site,
    team_score: u32,
    opponent_score: u32,
}

#[derive(PartialEq, Clone, Copy)]
enum Site {
    Home,
    Away,
    Neutral,
}

/// Parse the season game log.
///
/// Returns one `GameResult` per game, not per row.
pub fn parse_game_log(csv_text: &str) -> Result<Vec<GameResult>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(csv_text.as_bytes());

    let mut games = Vec::new();
    // Neutral-site games appear as two `N` rows with no home team to break the
    // tie, so they are deduplicated on the matchup itself.
    let mut seen_neutral: HashSet<(NaiveDate, String, String)> = HashSet::new();
    let mut skipped = 0usize;

    for (line, record) in reader.records().enumerate() {
        let record = record.map_err(|e| format!("barttorvik CSV line {}: {}", line + 1, e))?;
        let row = match parse_row(&record) {
            Some(row) => row,
            None => {
                skipped += 1;
                continue;
            }
        };

        let game = match row.site {
            // The away view of a game whose home view we will also see.
            Site::Away => continue,
            Site::Home => GameResult {
                game_id: game_id(&row.date, row.team, row.opponent),
                date: row.date,
                home_team_id: names::normalize(row.team),
                home_team_name: row.team.to_string(),
                away_team_id: names::normalize(row.opponent),
                away_team_name: row.opponent.to_string(),
                home_score: row.team_score,
                away_score: row.opponent_score,
                is_neutral_site: false,
                is_conference_game: row.team_conf == row.opponent_conf,
                is_completed: true,
            },
            Site::Neutral => {
                let mut pair = [row.team, row.opponent];
                pair.sort_unstable();
                let key = (
                    row.date,
                    names::normalize(pair[0]),
                    names::normalize(pair[1]),
                );
                if !seen_neutral.insert(key) {
                    continue;
                }
                GameResult {
                    game_id: game_id(&row.date, pair[0], pair[1]),
                    date: row.date,
                    home_team_id: names::normalize(row.team),
                    home_team_name: row.team.to_string(),
                    away_team_id: names::normalize(row.opponent),
                    away_team_name: row.opponent.to_string(),
                    home_score: row.team_score,
                    away_score: row.opponent_score,
                    is_neutral_site: true,
                    is_conference_game: row.team_conf == row.opponent_conf,
                    is_completed: true,
                }
            }
        };

        games.push(game);
    }

    if games.is_empty() {
        return Err(format!(
            "barttorvik returned no parsable games ({} rows skipped)",
            skipped
        ));
    }

    // A handful of unparsable rows is a postponed or in-progress game. A large
    // share of them means the column layout moved, which would otherwise show
    // up as quietly wrong ratings.
    let rows = games.len() * 2 + skipped;
    if skipped * 10 > rows {
        return Err(format!(
            "barttorvik: {} of {} rows did not parse — the CSV layout has probably changed",
            skipped, rows
        ));
    }

    games.sort_by(|a, b| a.date.cmp(&b.date).then_with(|| a.game_id.cmp(&b.game_id)));
    Ok(games)
}

fn parse_row<'a>(record: &'a csv::StringRecord) -> Option<Row<'a>> {
    if record.len() < MIN_COLUMNS {
        return None;
    }

    // `M/D/YY`, unpadded. Chrono accepts one or two digits for %m and %d.
    let date = NaiveDate::parse_from_str(record.get(COL_DATE)?.trim(), "%m/%d/%y").ok()?;

    let team = record.get(COL_TEAM)?.trim();
    let opponent = record.get(COL_OPPONENT)?.trim();
    if team.is_empty() || opponent.is_empty() {
        return None;
    }

    let site = match record.get(COL_SITE)?.trim() {
        "H" => Site::Home,
        "A" => Site::Away,
        "N" => Site::Neutral,
        _ => return None,
    };

    // "W, 96-62" / "L, 96-62" — the winner's score is always written first.
    let result = record.get(COL_RESULT)?.trim();
    let (outcome, scores) = result.split_once(',')?;
    let (winner_score, loser_score) = scores.trim().split_once('-')?;
    let winner_score: u32 = winner_score.trim().parse().ok()?;
    let loser_score: u32 = loser_score.trim().parse().ok()?;

    let (team_score, opponent_score) = match outcome.trim() {
        "W" => (winner_score, loser_score),
        "L" => (loser_score, winner_score),
        _ => return None,
    };

    Some(Row {
        date,
        team,
        team_conf: record.get(COL_CONF).unwrap_or("").trim(),
        opponent,
        opponent_conf: record.get(COL_OPP_CONF).unwrap_or("").trim(),
        site,
        team_score,
        opponent_score,
    })
}

fn game_id(date: &NaiveDate, a: &str, b: &str) -> String {
    format!("{}-{}-{}", date.format("%Y%m%d"), names::normalize(a), names::normalize(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two real games as the feed serves them: a home/away pair and a
    /// neutral-site pair, trimmed to the columns the parser reads.
    fn sample() -> String {
        let pad = |upto: usize, from: usize| ",".repeat(upto - from);
        let row = |date: &str, team: &str, conf: &str, opp: &str, site: &str, result: &str, oppconf: &str| {
            format!(
                "{},0,\"{}\",{},{},{},\"{}\"{}{}\n",
                date,
                team,
                conf,
                opp,
                site,
                result,
                pad(COL_OPP_CONF, COL_RESULT),
                oppconf
            )
        };
        let mut csv = String::new();
        csv.push_str(&row("12/16/25", "Abilene Christian", "WAC", "Arizona", "A", "L, 96-62", "B12"));
        csv.push_str(&row("12/16/25", "Arizona", "B12", "Abilene Christian", "H", "W, 96-62", "WAC"));
        csv.push_str(&row("11/24/25", "Abilene Christian", "WAC", "UTSA", "N", "W, 61-50", "AE"));
        csv.push_str(&row("11/24/25", "UTSA", "AE", "Abilene Christian", "N", "L, 61-50", "WAC"));
        csv
    }

    #[test]
    fn each_game_is_emitted_once() {
        let games = parse_game_log(&sample()).expect("parses");
        assert_eq!(games.len(), 2, "two games from four rows");
    }

    #[test]
    fn home_row_wins_the_pair_and_keeps_the_score_orientation() {
        let games = parse_game_log(&sample()).expect("parses");
        let game = games
            .iter()
            .find(|g| !g.is_neutral_site)
            .expect("the home/away pair");
        assert_eq!(game.home_team_name, "Arizona");
        assert_eq!(game.away_team_name, "Abilene Christian");
        assert_eq!(game.home_score, 96);
        assert_eq!(game.away_score, 62);
        assert!(game.home_won());
    }

    #[test]
    fn neutral_games_are_flagged_and_deduplicated() {
        let games = parse_game_log(&sample()).expect("parses");
        let neutral: Vec<_> = games.iter().filter(|g| g.is_neutral_site).collect();
        assert_eq!(neutral.len(), 1);
        // Whichever view survives, the winner has to be the winner.
        assert_eq!(neutral[0].winner_name(), "Abilene Christian");
        assert_eq!(neutral[0].margin(), 11);
    }

    #[test]
    fn a_loss_orients_the_score_toward_the_opponent() {
        let games = parse_game_log(&sample()).expect("parses");
        for game in &games {
            assert_ne!(game.home_score, game.away_score, "no ties in basketball");
        }
    }

    #[test]
    fn conference_games_are_detected_from_both_conference_columns() {
        let csv = format!(
            "1/8/26,0,\"Duke\",ACC,\"North Carolina\",H,\"W, 80-70\"{}{}\n",
            ",".repeat(COL_OPP_CONF - COL_RESULT),
            "ACC"
        );
        let games = parse_game_log(&csv).expect("parses");
        assert!(games[0].is_conference_game);
    }

    #[test]
    fn a_wholesale_layout_change_is_an_error_not_an_empty_season() {
        let err = parse_game_log("not,a,game,log\n").unwrap_err();
        assert!(err.contains("no parsable games"), "{}", err);
    }

    #[test]
    fn season_url_uses_the_ending_year() {
        assert!(season_url(2027).contains("year=2027"));
    }
}
