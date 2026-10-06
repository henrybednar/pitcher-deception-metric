"""
Pitcher Deception Project: swing-alignment outcomes (timing, horizontal alignment, whiff miss distance)
=========================================================================================================
Three regression outcomes built from Statcast bat tracking (see swing_alignment.py):

  timing     |depth deviation| from the hitter's ideal contact depth, on CONTACT swings (fouls and
             balls in play). Larger = the swing met the ball farther from the hitter's ideal depth,
             early or late.
  align      |reach deviation| from the hitter's ideal horizontal contact point, on contact swings.
             Tied up and flail both count.
  whiffmiss  log of Statcast miss_distance on whiffs. How badly the hitter missed.

Why contact only. On a whiff the intercept is the point of closest approach, which sits far from the
hitter's ideal contact point by construction: whiffs average 14.0 in of depth deviation against 7.3 in
on contact, and the whiff flag alone explains 16.7% of the variance. Scoring timing on all swings made
the pitcher-level timing residual correlate 0.39 with whiff rate; on contact swings it is 0.04. The
whiff side of the story is carried by whiff and by whiffmiss, so timing and alignment measure the
swings that made contact.

Each is fit at the full tier (fit_full_model.py) with GroupKFold by pitcher. The features also include
plate_x_inside, because depth and reach depend on whether the pitch was inside or outside to the
hitter. Previous-pitch features (pitch_pairs.py) are deliberately left out of every model. Sequencing
is something the pitcher chooses, so controlling for it would remove credit the metric is
meant to measure; sequencing_driver_analysis.py tests it as an explanation instead. Putting them in
the timing model once raised out-of-fold R2 by 0.003 (0.37% of squared error).

Writes the targets and predictions into per_pitch_predictions.csv by position, after checking the
row order matches.

MONTH RECALIBRATION: after the models (and the timing isotonic step below), each expectation is shifted by
the league-wide gap of its year and month, estimated from other pitchers only (fit_full_model.py has the
reasoning and the numbers).

TIMING RECALIBRATION: a calibration check found the timing model under-predicts actual deviation by
0.24 in at the low end of its predicted range (the other 9 deciles were within 0.08 in). Fixed with a
post-hoc, out-of-fold isotonic recalibration (recalibrate_oof_isotonic) rather than a different loss
function, since the miscalibration is confined to one tail rather than spread across the whole range.
"""

import pandas as pd
from sklearn.metrics import r2_score

from build_pitch_table import PITCH_LEVEL_FILE, load_pitch_data, read_aligned_predictions, save_predictions
from fit_full_model import FOLD_SEEDS, fit_full_outcome, prepare_context, recalibrate_oof_isotonic, recalibrate_scored

EXTRA_FEATURES = ["plate_x_inside"]
OUTCOMES = {
    "timing": dict(target="timing_dev_abs", subset="timing_mask", full_extra=EXTRA_FEATURES),
    "align": dict(target="align_dev_abs", subset="align_mask", full_extra=EXTRA_FEATURES),
    "whiffmiss": dict(target="whiffmiss_log", subset="whiffmiss_mask", full_extra=EXTRA_FEATURES),
}


def build_masks(df: pd.DataFrame) -> pd.DataFrame:
    contact = df["is_swing"] & (df["is_whiff"] == 0)
    df["timing_mask"] = contact & df["timing_dev_abs"].notna()
    df["align_mask"] = contact & df["align_dev_abs"].notna()
    df["whiffmiss_mask"] = (df["is_whiff"] == 1) & df["whiffmiss_log"].notna()
    return df


if __name__ == "__main__":
    print("loading full pitch-level data...", flush=True)
    df = build_masks(prepare_context(load_pitch_data(PITCH_LEVEL_FILE)))

    for name, spec in OUTCOMES.items():
        target = df.loc[df[spec["subset"]], spec["target"]]
        print(f"{name}: n={len(target):,}, mean={target.mean():.3f}, sd={target.std():.3f}, skew={target.skew():.2f}", flush=True)

    existing = read_aligned_predictions(df)

    new_cols = ["timing_dev", "timing_dev_abs", "align_dev", "align_dev_abs", "whiffmiss_log"]
    for label, spec in OUTCOMES.items():
        print(f"\n=== {label.upper()} (full tier: stuff+location+batter+catcher+context) ===", flush=True)
        fit_full_outcome(df, label, spec["target"], df[spec["subset"]], "regress", extra_features=spec["full_extra"],
                         fold_seeds=FOLD_SEEDS)
        new_cols.append(f"{label}_expected_full")

    # timing_mask alone still includes pitch types fit_full_outcome couldn't model (too rare, no
    # neighbor to pool with), which stay NaN — the same notna() guard every other outcome uses.
    timing_scored = df["timing_mask"] & df["timing_expected_full"].notna()
    before = df.loc[timing_scored, "timing_expected_full"].to_numpy()
    df.loc[timing_scored, "timing_expected_full"] = recalibrate_oof_isotonic(
        df.loc[timing_scored], "timing_dev_abs", "timing_expected_full")
    shift = df.loc[timing_scored, "timing_expected_full"].to_numpy() - before
    print(f"\ntiming recalibration: mean shift {shift.mean():+.4f} in, |shift| p95 {abs(shift).max():.4f} in "
          f"(R2={r2_score(df.loc[timing_scored, 'timing_dev_abs'], df.loc[timing_scored, 'timing_expected_full']):.4f})",
          flush=True)

    for label, spec in OUTCOMES.items():
        recalibrate_scored(df, label, spec["target"], binary=False)

    save_predictions(existing, df, new_cols)
