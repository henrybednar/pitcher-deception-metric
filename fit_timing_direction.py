"""
Pitcher Deception Project: signed timing direction (early vs. late)
====================================================================
The "timing" component models |depth deviation| from each hitter's ideal contact
depth (swing_alignment.py), so early and late contact both count as being fooled.
That makes it silent about direction. This fits the SIGNED deviation, in inches,
to show which way a pitcher pushes hitters: positive = contact farther out front
than the hitter's ideal (early), negative = deeper (late).

Sign check: the raw intercept depth correlated +0.71 with Savant's early_percent
leaderboard column and -0.72 with late_percent, so larger depth = early. (A one-off
check against swing_timing_by_pitcher_2025_2026.csv, joined on pitcher and season.)

Same architecture, features, and per-pitch-type grouping as every other outcome,
without the opponent and context features: stuff + location + plate_x_inside. The
result merges into per_pitch_predictions.csv by position after the row order is verified.

It is a readout, not a Deception+ component. Its expected value is the
stuff+location baseline, not the full tier.

CONTACT SWINGS ONLY, like the scored timing component. On a whiff the intercept is the point of closest
approach, which sits far from the hitter's ideal contact point by construction (fit_swing_alignment.py
has the numbers). The first version of this readout used every swing, and its pitcher-season value then
correlated 0.29 with the whiff index, 0.43 with whiff miss distance and 0.36 with chase.
"""

from build_pitch_table import PITCH_LEVEL_FILE, load_pitch_data, read_aligned_predictions, save_predictions
from fit_full_model import FOLD_SEEDS, LOCATION_FEATURES, STUFF_FEATURES, fit_tier
from fit_swing_alignment import build_masks

if __name__ == "__main__":
    print("loading full pitch-level data...", flush=True)
    df = build_masks(load_pitch_data(PITCH_LEVEL_FILE))
    df["timing_dir_raw"] = df["timing_dev"]
    print(f"total pitches: {len(df):,}", flush=True)
    existing = read_aligned_predictions(df)

    print("\n=== TIMING DIRECTION (signed, stuff+location) ===", flush=True)
    timing_mask = df["timing_mask"]
    fit_tier(df, "timingdir_expected_stuffloc", "timing_dir_raw", timing_mask, "regress",
             STUFF_FEATURES + LOCATION_FEATURES + ["plate_x_inside"], with_tendencies=False, fold_seeds=FOLD_SEEDS)
    save_predictions(existing, df, ["timing_dir_raw", "timingdir_expected_stuffloc"])
