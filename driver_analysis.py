"""
Pitcher Deception Project: driver analysis
==========================================
Regresses each component's scored residual (`<label>_diff_adj_shrunk` in the
canonical pitcher_season.csv) on the season-level driver features from
driver_features.py. The regression itself lives in driver_features.py
(run_driver_analysis): grouped-by-pitcher CV R^2, permutation importance and
Ridge coefficients.

The feature set excludes Savant's swing-outcome stats and a duplicated pace
column. driver_features.py explains why.

Run after add_timing_direction.py. Writes driver_analysis.json.
"""

import json

import pandas as pd

from driver_features import run_driver_analysis

COMPONENTS = ["whiff", "chase", "gb", "weak", "timing", "calledstrike", "align", "whiffmiss"]

if __name__ == "__main__":
    driver_df = pd.read_csv("output/driver_features.csv")
    print(f"driver features ({len(driver_df.columns) - 2}): "
          f"{[c for c in driver_df.columns if c not in ('pitcher', 'season')]}")

    ps = pd.read_csv("output/pitcher_season.csv")
    results = {label: run_driver_analysis(driver_df, ps, f"{label}_diff_adj_shrunk") for label in COMPONENTS}

    with open("output/driver_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved driver_analysis.json. Done.")
