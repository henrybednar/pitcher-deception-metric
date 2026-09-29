"""
Pitcher Deception Project: export leaderboard_data.json from the canonical pitcher_season.csv
==============================================================================================
build_leaderboard.py only reads this JSON. It holds every scored pitcher-season with the
component indexes, sample sizes, intervals, the role proxy from export_site_stats.pitcher_roles(), and the
signed timing readout in inches (timing_bias_inches). Ground ball is scored and shown but is not
part of Deception+. See COMPOSITE_OUTCOMES in reliability_and_ci.py.
"""

import json

import pandas as pd

from export_site_stats import pitcher_roles

ps = pd.read_csv("output/pitcher_season.csv")
raw = pd.read_csv("raw/statcast_pitch_level_2025_2026.csv", usecols=["pitcher", "season", "game_pk"], low_memory=False)
ps = ps.merge(pitcher_roles(raw), on=["pitcher", "season"], how="left")

cols = [
    "player_name", "season", "qualified", "deception_plus",
    "whiff_index", "whiff_n", "whiff_ci_lo", "whiff_ci_hi",
    "chase_index", "chase_n",
    "gb_index", "gb_n",
    "weak_index", "weak_n",
    "timing_index", "timing_n",
    "whiffmiss_index", "whiffmiss_n", "whiffmiss_ci_lo", "whiffmiss_ci_hi",
    "timing_bias_inches", "timing_bias_ci_lo", "timing_bias_ci_hi", "timingdir_n",
    "stuff_plus", "role", "med_pitches_per_app",
]
out = ps.dropna(subset=["deception_plus"])[cols].rename(columns={"player_name": "name"})

INT_COLS = {"season", "whiff_n", "chase_n", "gb_n", "weak_n", "timing_n", "timingdir_n",
            "whiffmiss_n", "stuff_plus"}
INCH_COLS = {"timing_bias_inches", "timing_bias_ci_lo", "timing_bias_ci_hi"}
for c in out.columns:
    if c in INT_COLS:
        out[c] = out[c].round(0)
    elif c in INCH_COLS:
        out[c] = out[c].round(2)
    elif pd.api.types.is_float_dtype(out[c]):
        out[c] = out[c].round(1)

rows = json.loads(out.to_json(orient="records"))

with open("output/leaderboard_data.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, separators=(",", ":"))

print(f"Saved leaderboard_data.json: {len(rows):,} rows, {len(cols)} fields each.")
