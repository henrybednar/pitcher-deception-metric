"""
Pitcher Deception Project: export leaderboard_data.json from the canonical pitcher_season.csv
==============================================================================================
build_leaderboard.py only reads this JSON. It holds every scored pitcher-season with the
component indexes, sample sizes, 95% intervals for Deception+ and every component, the next-season projection with its 80% range
and basis (projection.py), the role label from export_site_stats.pitcher_roles(), and the
signed timing readout in inches (timing_bias_inches), and each row's whiff and chase scores by pitch type
(`types`, from pitch_type_scores.py). Ground ball is scored and shown but is not
part of Deception+. See COMPOSITE_OUTCOMES in reliability_and_ci.py.
"""

import json

import pandas as pd

from export_site_stats import pitcher_roles
from pitch_type_scores import leaderboard_lists

ps = pd.read_csv("output/pitcher_season.csv")
raw = pd.read_csv("raw/statcast_pitch_level_2024_2026.csv", usecols=["pitcher", "season", "game_pk", "game_type"], low_memory=False)
raw = raw[raw["game_type"] == "R"]
ps = ps.merge(pitcher_roles(raw, usage=ps[["pitcher", "season", "games", "games_started"]]),
              on=["pitcher", "season"], how="left")
ps = ps.merge(pd.read_csv("output/projection.csv").rename(columns={"projection": "proj", "proj_lo": "proj_lo", "proj_hi": "proj_hi"}),
              on=["pitcher", "season"], how="left")

cols = [
    "player_name", "season", "qualified", "deception_plus", "deception_plus_ci_lo", "deception_plus_ci_hi",
    "whiff_index", "whiff_n", "whiff_ci_lo", "whiff_ci_hi",
    "chase_index", "chase_n", "chase_ci_lo", "chase_ci_hi",
    "gb_index", "gb_n", "gb_ci_lo", "gb_ci_hi",
    "weak_index", "weak_n", "weak_ci_lo", "weak_ci_hi",
    "timing_index", "timing_n", "timing_ci_lo", "timing_ci_hi",
    "whiffmiss_index", "whiffmiss_n", "whiffmiss_ci_lo", "whiffmiss_ci_hi",
    "timing_bias_inches", "timing_bias_ci_lo", "timing_bias_ci_hi", "timingdir_n",
    "stuff_plus", "role", "med_pitches_per_app", "proj", "proj_lo", "proj_hi", "proj_basis",
]
scored = ps.dropna(subset=["deception_plus"])
keys = list(zip(scored["pitcher"].astype(int), scored["season"].astype(int)))
out = scored[cols].rename(columns={"player_name": "name"})

INT_COLS = {"season", "whiff_n", "chase_n", "gb_n", "weak_n", "timing_n", "timingdir_n",
            "whiffmiss_n", "stuff_plus", "proj_basis"}
INCH_COLS = {"timing_bias_inches", "timing_bias_ci_lo", "timing_bias_ci_hi"}
for c in out.columns:
    if c in INT_COLS:
        out[c] = out[c].round(0)
    elif c in INCH_COLS:
        out[c] = out[c].round(2)
    elif pd.api.types.is_float_dtype(out[c]):
        out[c] = out[c].round(1)

rows = json.loads(out.to_json(orient="records"))
types = leaderboard_lists(pd.read_csv("output/pitch_type_scores.csv"))
for row, key in zip(rows, keys):
    row["types"] = types.get(key, [])

with open("output/leaderboard_data.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, separators=(",", ":"), allow_nan=False)

print(f"Saved leaderboard_data.json: {len(rows):,} rows, {len(cols) + 1} fields each.")
