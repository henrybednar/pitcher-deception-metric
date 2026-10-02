"""
Pitcher Deception Project — add the timing-direction diagnostic to pitcher_season.csv
=========================================================================================
Reuses reliability_and_ci.py's machinery (game-level aggregation, empirical-Bayes
shrinkage, bootstrap CI, split-half reliability) on the "timing_dir_raw" outcome
from fit_timing_direction.py, and reports it in raw inches instead of a 100/10
index. Positive = contact farther out front than the hitter's ideal depth and than
stuff+location predicts (earlier), negative = deeper (later). The sign is confirmed
in fit_timing_direction.py against Savant's early_percent and late_percent columns.

The timing component scores the size of the deviation and this column scores its
direction, so a pitcher can be high on one and near zero on the other.
"""

import pandas as pd

from reliability_and_ci import game_level_table, pitcher_season_point_estimate, process_outcome, shrink_and_scale

if __name__ == "__main__":
    print("loading per-pitch predictions...", flush=True)
    df = pd.read_csv("output/per_pitch_predictions.csv")

    spec = dict(subset=lambda d: d["is_swing"] & (d["is_whiff"] == 0) & d["timing_dir_raw"].notna(), target="timing_dir_raw")
    result, rel, half_tbl = process_outcome(df, "timingdir", spec, is_binary=False, baseline="stuffloc")

    # process_outcome's own ci_lo/ci_hi are on the 100/10 "index" scale (same
    # transform as every other component) — but timing_bias_inches is
    # deliberately reported in raw inches, not that scale, so the CI has to
    # be converted back rather than reused as-is (the forward transform is
    # linear and monotonic, so this is an exact inverse, not an approximation).
    mask = spec["subset"](df) & df["timingdir_expected_stuffloc"].notna()
    sub = df[mask].copy()
    resid = sub["timing_dir_raw"] - sub["timingdir_expected_stuffloc"]
    sub["_var_contrib"] = (resid ** 2).groupby(sub["pitch_type"]).transform("mean")
    full_games = game_level_table(sub, "timingdir", "timing_dir_raw", baseline="stuffloc", var_col="_var_contrib")
    full_agg = pitcher_season_point_estimate(full_games)
    # Reuse process_outcome's own design_effect (not a fresh estimate) so
    # this recomputed league_std is exactly consistent with the ci_lo/ci_hi
    # values it's about to invert — a different estimate here would silently
    # misconvert the CI back to inches.
    _, _, league_std, *_ = shrink_and_scale(full_agg, rel["design_effect"])

    result = result.rename(columns={"timingdir_diff_adj_shrunk": "timing_bias_inches"})
    result["timing_bias_ci_lo"] = (result["timingdir_ci_lo"] - 100) * league_std / 10
    result["timing_bias_ci_hi"] = (result["timingdir_ci_hi"] - 100) * league_std / 10
    keep = ["pitcher", "season", "timingdir_n", "timing_bias_inches", "timing_bias_ci_lo", "timing_bias_ci_hi"]
    result = result[keep]

    ps = pd.read_csv("output/pitcher_season.csv")
    ps = ps.drop(columns=[c for c in keep if c in ps.columns and c not in ("pitcher", "season")], errors="ignore")
    ps = ps.merge(result, on=["pitcher", "season"], how="left")
    ps.to_csv("output/pitcher_season.csv", index=False)

    rr = pd.read_csv("output/reliability_report.csv")
    rr = rr[rr["label"] != "timingdir"]
    rr = pd.concat([rr, pd.DataFrame([rel])], ignore_index=True)
    rr.to_csv("output/reliability_report.csv", index=False)

    print(f"\ntiming_bias_inches: mean={ps['timing_bias_inches'].mean():.3f}, "
          f"n_scored={ps['timing_bias_inches'].notna().sum()}", flush=True)
    print("Saved pitcher_season.csv (+timing_bias_inches) and reliability_report.csv. Done.", flush=True)
