# Win-probability backtest harness

Throwaway Python used to produce `docs/WIN_PROBABILITY_METHODS.md`. It is not
part of the Rust build and nothing in `src/` depends on it. It lives here so the
numbers in that document can be re-derived rather than taken on faith.

## Data

Nine seasons, 49,053 games, 600 NCAA tournament games (2017-2019, 2021-2026;
2020 has no tournament and is used only as a carryover source, as is 2016).

- `https://barttorvik.com/getgamestats.php?year=YYYY&csv=1` — one request per
  season, every D1 game with date, site (H/A/N), score, tempo and four factors.
- `https://ncaa-api.henrygd.me/brackets/basketball-men/d1/YYYY` — the bracket,
  used only to label which games were tournament games (matched on date + score
  pair, so no name resolution is involved).
- `https://barttorvik.com/trank.php?year=YYYY&begin=...&end=...&csv=1` — Torvik's
  own ratings as of the day before the tournament, used as an external
  benchmark. This endpoint sits behind a JS check; a `POST js_test_submitted=1`
  sets the cookie that lets a subsequent `GET` through.

```bash
pip install numpy pandas scikit-learn lightgbm
mkdir -p tv br trank
# fetch the CSVs listed above into tv/, br/, trank/
python3 prep.py          # builds games.pkl with tournament labels
python3 exp_gbdt.py      # builds feats.pkl (slow: ~15 min)
python3 final_table.py   # the headline table
```

## Protocol

Every prediction is walk-forward: a game is scored with ratings built only from
games that finished before it. The rating-to-probability link is a two-parameter
logistic fit **leave-one-season-out** — fitted on eight seasons, applied to the
ninth — so the calibration is out of sample too.

Three evaluation populations are reported separately: `early` (first 45 days),
`late` (day 45 onward), and `tourney` (the 600 tournament games). Do not compare
a tournament number to a season number; the tournament truncates both tails of
the matchup distribution and its log loss is not on the same scale.
