//! Deterministic team-name resolution.
//!
//! Ratings come from ESPN/NCAA/538, bracket entries come from a different feed,
//! and the two spell schools differently. Resolving one to the other used to be
//! a `.contains()` scan over a `HashMap`, which meant "Texas" could resolve to
//! "Texas A&M" or "Texas Tech" depending on hash iteration order — a different
//! answer on different runs, with no warning.
//!
//! This module normalizes both sides, tries progressively looser matches, and
//! refuses to guess when more than one candidate survives a tier.

use std::collections::BTreeSet;

/// Whole-string aliases, applied after normalization. Only exact normalized
/// matches are rewritten, so "unc" maps to North Carolina while "uncw"
/// (UNC Wilmington) is left alone.
const ALIASES: &[(&str, &str)] = &[
    ("uconn", "connecticut"),
    ("unc", "north carolina"),
    ("ncstate", "north carolina state"),
    ("ncst", "north carolina state"),
    ("usc", "southern california"),
    ("ucf", "central florida"),
    ("smu", "southern methodist"),
    ("vcu", "virginia commonwealth"),
    ("lsu", "louisiana state"),
    ("byu", "brigham young"),
    ("tcu", "texas christian"),
    ("utep", "texas el paso"),
    ("utsa", "texas san antonio"),
    ("uab", "alabama birmingham"),
    ("umbc", "maryland baltimore county"),
    ("olemiss", "mississippi"),
    ("pitt", "pittsburgh"),
    ("umass", "massachusetts"),
    ("smiss", "southern mississippi"),
    ("liu", "long island"),
    ("liubrooklyn", "long island"),
];

/// Lowercase, expand the `St`/`St.` abbreviation by position, and strip
/// everything that is not a letter or digit.
///
/// `St` leading a name means "Saint" (St. John's); `St` trailing a name means
/// "State" (Michigan St). Getting this backwards merges unrelated schools, so
/// the two cases are handled separately rather than with a blanket rewrite.
pub fn normalize(name: &str) -> String {
    let lowered = name.to_lowercase();

    let tokens: Vec<String> = lowered
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect();

    if tokens.is_empty() {
        return String::new();
    }

    let last = tokens.len() - 1;
    let expanded: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, token)| match token.as_str() {
            "st" | "ste" if i == 0 => "saint".to_string(),
            "st" if i == last => "state".to_string(),
            "univ" | "university" => String::new(),
            "the" if i == 0 => String::new(),
            other => other.to_string(),
        })
        .filter(|t| !t.is_empty())
        .collect();

    let joined = expanded.join(" ");
    let collapsed: String = joined.chars().filter(|c| c.is_alphanumeric()).collect();

    for (alias, canonical) in ALIASES {
        if collapsed == *alias {
            return canonical.chars().filter(|c| c.is_alphanumeric()).collect();
        }
    }

    collapsed
}

/// Why a name could not be resolved to exactly one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameError {
    NotFound {
        query: String,
        suggestions: Vec<String>,
    },
    Ambiguous {
        query: String,
        matches: Vec<String>,
    },
}

impl std::fmt::Display for NameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NameError::NotFound { query, suggestions } if suggestions.is_empty() => {
                write!(f, "no team matches '{}'", query)
            }
            NameError::NotFound { query, suggestions } => write!(
                f,
                "no team matches '{}'; did you mean one of: {}?",
                query,
                suggestions.join(", ")
            ),
            NameError::Ambiguous { query, matches } => write!(
                f,
                "'{}' is ambiguous — it matches {}. Use the full name.",
                query,
                matches.join(", ")
            ),
        }
    }
}

impl std::error::Error for NameError {}

/// Resolve `query` against `candidates`, returning the index of the single match.
///
/// Tiers are tried in order and the first tier that produces any match decides
/// the outcome: exactly one match resolves, more than one is an error. A looser
/// tier is never used to break a tie in a tighter one.
///
/// 1. exact normalized equality
/// 2. one normalized name is a prefix of the other
/// 3. one normalized name contains the other
pub fn resolve(query: &str, candidates: &[String]) -> Result<usize, NameError> {
    let needle = normalize(query);
    if needle.is_empty() {
        return Err(NameError::NotFound {
            query: query.to_string(),
            suggestions: Vec::new(),
        });
    }

    let normalized: Vec<String> = candidates.iter().map(|c| normalize(c)).collect();

    let tiers: [fn(&str, &str) -> bool; 3] = [
        |a, b| a == b,
        |a, b| a.starts_with(b) || b.starts_with(a),
        |a, b| a.contains(b) || b.contains(a),
    ];

    for matches_tier in tiers {
        let hits: Vec<usize> = normalized
            .iter()
            .enumerate()
            .filter(|(_, cand)| matches_tier(cand, &needle))
            .map(|(i, _)| i)
            .collect();

        match hits.len() {
            0 => continue,
            1 => return Ok(hits[0]),
            _ => {
                // Sorted so the error message is stable run to run.
                let names: BTreeSet<String> =
                    hits.iter().map(|&i| candidates[i].clone()).collect();
                return Err(NameError::Ambiguous {
                    query: query.to_string(),
                    matches: names.into_iter().collect(),
                });
            }
        }
    }

    // Nothing matched at any tier: offer names sharing a leading run of characters.
    let mut scored: Vec<(usize, &String)> = candidates
        .iter()
        .enumerate()
        .map(|(i, name)| (shared_prefix_len(&normalized[i], &needle), name))
        .filter(|(len, _)| *len >= 3)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

    Err(NameError::NotFound {
        query: query.to_string(),
        suggestions: scored.into_iter().take(3).map(|(_, n)| n.clone()).collect(),
    })
}

fn shared_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn teams() -> Vec<String> {
        [
            "Texas",
            "Texas A&M",
            "Texas Tech",
            "Connecticut",
            "North Carolina",
            "North Carolina State",
            "Michigan St",
            "St. John's",
            "Saint Mary's",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn exact_match_wins_over_substrings() {
        // The old substring scan resolved this to whichever of the three the
        // hash iterator reached first.
        let t = teams();
        assert_eq!(t[resolve("Texas", &t).unwrap()], "Texas");
        assert_eq!(t[resolve("Texas Tech", &t).unwrap()], "Texas Tech");
        assert_eq!(
            t[resolve("North Carolina", &t).unwrap()],
            "North Carolina"
        );
    }

    #[test]
    fn ambiguity_is_an_error_not_a_guess() {
        let t = teams();
        // "Texas A" extends "Texas" and is extended by "Texas A&M"; neither is a
        // better answer than the other, so this must not silently pick one.
        match resolve("Texas A", &t).unwrap_err() {
            NameError::Ambiguous { matches, .. } => {
                assert_eq!(matches, vec!["Texas".to_string(), "Texas A&M".to_string()]);
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn genuinely_ambiguous_prefixes_are_rejected() {
        let t = vec!["Miami (FL)".to_string(), "Miami (OH)".to_string()];
        assert!(matches!(
            resolve("Miami", &t),
            Err(NameError::Ambiguous { .. })
        ));
    }

    #[test]
    fn aliases_resolve() {
        let t = teams();
        assert_eq!(t[resolve("UConn", &t).unwrap()], "Connecticut");
        assert_eq!(t[resolve("UNC", &t).unwrap()], "North Carolina");
        assert_eq!(t[resolve("NC State", &t).unwrap()], "North Carolina State");
    }

    #[test]
    fn saint_and_state_abbreviations_go_opposite_ways() {
        assert_eq!(normalize("St. John's"), normalize("Saint Johns"));
        assert_eq!(normalize("Michigan St"), normalize("Michigan State"));
        assert_ne!(normalize("St. John's"), normalize("John State"));
    }

    #[test]
    fn resolution_is_order_independent() {
        let mut t = teams();
        let first = t[resolve("Texas", &t).unwrap()].clone();
        t.reverse();
        let second = t[resolve("Texas", &t).unwrap()].clone();
        assert_eq!(first, second);
    }

    #[test]
    fn missing_names_suggest_rather_than_guess() {
        let t = teams();
        match resolve("Connecticutt Huskies", &t) {
            Ok(i) => assert_eq!(t[i], "Connecticut"),
            Err(NameError::NotFound { suggestions, .. }) => {
                assert!(suggestions.contains(&"Connecticut".to_string()))
            }
            Err(e) => panic!("unexpected: {:?}", e),
        }
        assert!(matches!(
            resolve("Nowhere Polytechnic", &t),
            Err(NameError::NotFound { .. })
        ));
    }
}
