//! The tournament field, from the NCAA's own published bracket.
//!
//! `ncaa-api.henrygd.me/brackets/basketball-men/d1/{year}` mirrors the bracket
//! behind ncaa.com: all 67 games, each with a `bracketPositionId` that encodes
//! its round and slot, and each team carrying its seed. It is free, needs no
//! key, and is authoritative — which the alternative was not. Reconstructing
//! the field by scraping tournament scoreboards guessed regions out of the
//! headline text of game notes, and could not produce a bracket at all until
//! the games had been scheduled.
//!
//! Position ids are `RSS`: round digit, then slot. `2xx` is the round of 64,
//! `3xx` the round of 32, up to `701` for the final. Sections 2-5 are the four
//! regions; section 1 is the First Four and section 6 is the Final Four.

use crate::game_result::BracketTeam;
use crate::names;
use serde_json::Value;

/// First and last `bracketPositionId` of the round of 64.
const R64_FIRST: i64 = 201;
const R64_LAST: i64 = 232;
/// The two national semifinals and the final.
const SEMIFINALS: [i64; 2] = [601, 602];
const FINAL: i64 = 701;

/// A tournament field, with the region pairing the bracket actually uses.
#[derive(Debug, Clone)]
pub struct BracketField {
    pub teams: Vec<BracketTeam>,
    /// The four regions in bracket order: `[0]` meets `[1]` in one semifinal,
    /// `[2]` meets `[3]` in the other.
    ///
    /// This is not fixed across years — the NCAA rotates it. In 2026 the East
    /// met the South and the West met the Midwest; assuming any particular
    /// pairing decides which teams can ever meet, and so which bracket the
    /// optimizer thinks is best.
    pub region_layout: [String; 4],
}

pub fn bracket_url(year: i32) -> String {
    format!(
        "https://ncaa-api.henrygd.me/brackets/basketball-men/d1/{}",
        year
    )
}

/// Title-case the feed's shouted region names into the spelling the rest of
/// the program uses: `" MIDWEST"` becomes `"Midwest"`.
fn canonical_region(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(trimmed.len());
    for (i, ch) in trimmed.chars().enumerate() {
        if i == 0 {
            out.extend(ch.to_uppercase());
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    Some(out)
}

pub fn parse_bracket(json: &Value, year: i32) -> Result<BracketField, String> {
    let championship = json
        .get("championships")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .ok_or_else(|| format!("no championship in the {} bracket feed", year))?;

    let games = championship
        .get("games")
        .and_then(|g| g.as_array())
        .ok_or_else(|| format!("no games in the {} bracket feed", year))?;

    // sectionId -> region name, from the feed's own region list.
    let mut region_of_section: Vec<(i64, String)> = Vec::new();
    for region in championship
        .get("regions")
        .and_then(|r| r.as_array())
        .unwrap_or(&Vec::new())
    {
        let section = match region.get("sectionId").and_then(|s| s.as_i64()) {
            Some(section) => section,
            None => continue,
        };
        let title = region.get("title").and_then(|t| t.as_str()).unwrap_or("");
        if let Some(name) = canonical_region(title) {
            region_of_section.push((section, name));
        }
    }

    if region_of_section.len() != 4 {
        return Err(format!(
            "expected 4 named regions in the {} bracket, found {}. \
             The bracket is probably not published yet.",
            year,
            region_of_section.len()
        ));
    }

    let mut teams: Vec<BracketTeam> = Vec::with_capacity(64);
    let mut unresolved: Vec<String> = Vec::new();

    for game in games {
        let position = match game.get("bracketPositionId").and_then(|p| p.as_i64()) {
            Some(position) => position,
            None => continue,
        };
        if !(R64_FIRST..=R64_LAST).contains(&position) {
            continue;
        }

        let section = game
            .get("sectionId")
            .and_then(|s| s.as_i64())
            .ok_or_else(|| format!("game {} has no section", position))?;
        let region = region_of_section
            .iter()
            .find(|(id, _)| *id == section)
            .map(|(_, name)| name.clone())
            .ok_or_else(|| format!("game {} is in unknown section {}", position, section))?;

        let sides = game
            .get("teams")
            .and_then(|t| t.as_array())
            .ok_or_else(|| format!("game {} has no teams", position))?;

        for side in sides {
            let seed = side.get("seed").and_then(|s| s.as_i64()).unwrap_or(0) as i32;
            let name = side
                .get("nameShort")
                .and_then(|n| n.as_str())
                .filter(|n| !n.trim().is_empty())
                .or_else(|| side.get("nameFull").and_then(|n| n.as_str()));

            match (name, seed) {
                (Some(name), 1..=16) => teams.push(BracketTeam::new(
                    names::normalize(name),
                    name.trim().to_string(),
                    seed,
                    region.clone(),
                )),
                // A First Four slot that has not been decided yet: the feed
                // carries a placeholder rather than a school.
                _ => unresolved.push(format!(
                    "  {} seed in the {} (round-of-64 game {})",
                    if seed > 0 {
                        seed.to_string()
                    } else {
                        "unknown".to_string()
                    },
                    region,
                    position
                )),
            }
        }
    }

    if !unresolved.is_empty() {
        return Err(format!(
            "{} slot(s) in the {} bracket are not filled in yet:\n{}\n\
             First Four winners are not known until Wednesday night. Rerun \
             after they are played, or supply the field with --bracket-file.",
            unresolved.len(),
            year,
            unresolved.join("\n")
        ));
    }

    if teams.len() != 64 {
        return Err(format!(
            "expected 64 teams in the {} bracket, parsed {}",
            year,
            teams.len()
        ));
    }

    let region_layout = read_region_layout(games, &region_of_section, year)?;

    Ok(BracketField {
        teams,
        region_layout,
    })
}

/// Work out which regions meet in which semifinal.
///
/// Each regional final names the semifinal it feeds and which half of it, so
/// the pairing is read off the bracket rather than assumed.
fn read_region_layout(
    games: &[Value],
    region_of_section: &[(i64, String)],
    year: i32,
) -> Result<[String; 4], String> {
    // (semifinal position, half rank, region). "Top" sorts before "Bottom" by
    // rank rather than alphabetically, which would reverse them.
    let mut slots: Vec<(i64, u8, String)> = Vec::new();

    for game in games {
        let position = match game.get("bracketPositionId").and_then(|p| p.as_i64()) {
            Some(position) => position,
            None => continue,
        };
        // Regional finals: round 5, one per region.
        if !(501..=599).contains(&position) {
            continue;
        }
        let victor = game
            .get("victorBracketPositionId")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if !SEMIFINALS.contains(&victor) && victor != FINAL {
            continue;
        }
        let half = match game.get("victorGamePosition").and_then(|v| v.as_str()) {
            Some("Top") => 0,
            Some("Bottom") => 1,
            _ => continue,
        };
        let section = game.get("sectionId").and_then(|s| s.as_i64()).unwrap_or(-1);
        if let Some((_, region)) = region_of_section.iter().find(|(id, _)| *id == section) {
            slots.push((victor, half, region.clone()));
        }
    }

    if slots.len() != 4 {
        return Err(format!(
            "could not read the {} Final Four pairing: found {} regional finals",
            year,
            slots.len()
        ));
    }

    // Order: first semifinal's two regions, then the second semifinal's.
    slots.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let layout: Vec<String> = slots.into_iter().map(|(_, _, region)| region).collect();

    let mut sorted = layout.clone();
    sorted.sort();
    sorted.dedup();
    if sorted.len() != 4 {
        return Err(format!(
            "the {} Final Four pairing names a region twice: {:?}",
            year, layout
        ));
    }

    Ok([
        layout[0].clone(),
        layout[1].clone(),
        layout[2].clone(),
        layout[3].clone(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A bracket in the shape the feed serves, with the 2026 region pairing:
    /// East/South in one semifinal, West/Midwest in the other.
    fn feed() -> Value {
        let regions: Vec<Value> = [(2, " EAST"), (3, "WEST"), (4, " SOUTH"), (5, " MIDWEST")]
            .iter()
            .map(|(section, title)| json!({"sectionId": section, "title": title}))
            .collect();

        let mut games: Vec<Value> = Vec::new();
        let matchups = [
            (1, 16),
            (8, 9),
            (5, 12),
            (4, 13),
            (6, 11),
            (3, 14),
            (7, 10),
            (2, 15),
        ];
        // Round of 64: eight games per region, sections in position order.
        for (offset, section) in [(0, 2), (8, 4), (16, 3), (24, 5)] {
            for (slot, (top, bottom)) in matchups.iter().enumerate() {
                games.push(json!({
                    "bracketPositionId": 201 + offset + slot as i64,
                    "sectionId": section,
                    "teams": [
                        {"seed": top, "nameShort": format!("S{} of {}", top, section)},
                        {"seed": bottom, "nameShort": format!("S{} of {}", bottom, section)},
                    ]
                }));
            }
        }
        // Regional finals and the pairing they feed.
        for (position, section, victor, half) in [
            (501, 2, 601, "Top"),
            (502, 4, 601, "Bottom"),
            (503, 3, 602, "Top"),
            (504, 5, 602, "Bottom"),
        ] {
            games.push(json!({
                "bracketPositionId": position,
                "sectionId": section,
                "victorBracketPositionId": victor,
                "victorGamePosition": half,
                "teams": []
            }));
        }

        json!({"championships": [{"regions": regions, "games": games}]})
    }

    #[test]
    fn parses_all_sixty_four_teams() {
        let field = parse_bracket(&feed(), 2026).expect("parses");
        assert_eq!(field.teams.len(), 64);
        for region in ["East", "West", "South", "Midwest"] {
            let count = field.teams.iter().filter(|t| t.region == region).count();
            assert_eq!(count, 16, "{} should have 16 teams", region);
        }
    }

    #[test]
    fn every_seed_appears_once_per_region() {
        let field = parse_bracket(&feed(), 2026).expect("parses");
        for region in ["East", "West", "South", "Midwest"] {
            let mut seeds: Vec<i32> = field
                .teams
                .iter()
                .filter(|t| t.region == region)
                .map(|t| t.seed)
                .collect();
            seeds.sort_unstable();
            assert_eq!(seeds, (1..=16).collect::<Vec<_>>(), "seeds in {}", region);
        }
    }

    #[test]
    fn shouted_region_names_are_normalized() {
        let field = parse_bracket(&feed(), 2026).expect("parses");
        assert!(field.teams.iter().all(|t| t.region != " MIDWEST"));
        assert!(field.teams.iter().any(|t| t.region == "Midwest"));
    }

    #[test]
    fn the_final_four_pairing_is_read_from_the_bracket() {
        let field = parse_bracket(&feed(), 2026).expect("parses");
        assert_eq!(
            field.region_layout,
            [
                "East".to_string(),
                "South".to_string(),
                "West".to_string(),
                "Midwest".to_string()
            ],
            "2026 paired East with South, not the alphabetical default"
        );
    }

    #[test]
    fn an_undecided_first_four_slot_is_an_error_not_a_silent_hole() {
        let mut feed = feed();
        feed["championships"][0]["games"][0]["teams"][1] =
            json!({"seed": 16, "nameShort": "", "nameFull": null});
        let err = parse_bracket(&feed, 2026).unwrap_err();
        assert!(err.contains("not filled in yet"), "{}", err);
    }

    #[test]
    fn a_bracket_that_is_not_published_yet_says_so() {
        let empty = json!({"championships": [{"regions": [], "games": []}]});
        let err = parse_bracket(&empty, 2027).unwrap_err();
        assert!(err.contains("not published yet"), "{}", err);
    }
}
