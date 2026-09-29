"""
Pitcher Deception Project: swing alignment against each hitter's ideal contact point
=====================================================================================
Statcast's per-pitch bat tracking gives two intercept measures, both taken where the
bat meets the ball, relative to the batter's center of mass:

  intercept_ball_minus_batter_pos_y_inches   depth, mound-to-plate axis (about 30 in
                                             in front of the body; larger = farther
                                             out front, which is earlier contact)
  intercept_ball_minus_batter_pos_x_inches   reach, horizontal axis (about 37 in;
                                             small = tied up, large = flailing)

Neither is an error. Savant's early/late and tied-up/flail categories are measured
against "the hitter's ideal contact point", which the pitch-level export does not
include. This module estimates it and turns each swing into a deviation.

Ideal contact point of a hitter, per season
  1. Score each ball in play by contact quality: launch speed minus what bat speed
     and pitch speed predict (OLS across all balls in play). This removes the fact
     that harder pitches and faster swings produce harder contact for free.
  2. Take the hitter's top quartile by that score and average their depth and reach.
  3. Leave out every ball put in play against the pitcher being scored
     (leave-one-PITCHER-out, the same rule as the batter and catcher tendencies), so
     a pitcher's own pitches never set the reference that scores them.
  4. Shrink toward the league value with SHRINK_K pseudo-balls, so a hitter with few
     hard-hit balls falls back to the league ideal.

Per-swing outputs (NaN when the intercept is missing):
  timing_dev       signed depth deviation, inches. Positive = early, negative = late.
  timing_dev_abs   absolute depth deviation. The scored timing component (on contact swings).
  align_dev        signed reach deviation, inches. Negative = tied up, positive = flail.
  align_dev_abs    absolute reach deviation. The horizontal alignment component.
  whiffmiss_log    log of Statcast miss_distance, whiffs only (bat-to-ball distance at
                   closest approach).
  plate_x_inside   plate_x flipped so positive means inside to the hitter (NaN when the batting
                   side is missing). Depth and
                   reach depend on inside or outside location, and the pitcher-arm-side
                   plate_x used elsewhere does not carry that.
"""

import numpy as np
import pandas as pd

TOP_QUANTILE = 0.75
SHRINK_K = 10
DEPTH_COL = "intercept_ball_minus_batter_pos_y_inches"
REACH_COL = "intercept_ball_minus_batter_pos_x_inches"
KEYS = ["batter", "season"]


def finite(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """True where a value is a real number. notna() alone lets inf through, and one inf in a
    least-squares input or a group mean makes every downstream value non-finite."""
    return np.isfinite(frame.astype(float))


def contact_quality(df: pd.DataFrame) -> pd.Series:
    """Launch speed minus the OLS prediction from bat speed and pitch speed, on balls in play."""
    fields = ["launch_speed", "bat_speed", "release_speed"]
    rows = df["is_bip"] & finite(df[fields]).all(axis=1)
    design = np.column_stack([np.ones(int(rows.sum())), df.loc[rows, "bat_speed"], df.loc[rows, "release_speed"]])
    coef, *_ = np.linalg.lstsq(design, df.loc[rows, "launch_speed"].to_numpy(float), rcond=None)
    quality = pd.Series(np.nan, index=df.index)
    quality.loc[rows] = df.loc[rows, "launch_speed"].to_numpy(float) - design @ coef
    return quality


def leave_pitcher_out_ideal(df: pd.DataFrame, top: pd.Series, value_col: str) -> np.ndarray:
    """Per-row ideal value: mean of value_col over the hitter's top-quality balls in play,
    excluding this row's pitcher, shrunk toward the league mean of those balls."""
    top_rows = df.loc[top, KEYS + ["pitcher", value_col]]
    if top_rows.empty:
        raise ValueError(f"no top-quality balls in play with a finite {value_col}; cannot estimate an ideal contact point")
    league = float(top_rows[value_col].mean())

    by_hitter = top_rows.groupby(KEYS)[value_col].agg(hitter_sum="sum", hitter_n="count").reset_index()
    by_pair = top_rows.groupby(KEYS + ["pitcher"])[value_col].agg(pair_sum="sum", pair_n="count").reset_index()

    keyed = df[KEYS + ["pitcher"]].merge(by_hitter, on=KEYS, how="left").merge(by_pair, on=KEYS + ["pitcher"], how="left")
    hitter_sum, hitter_n = keyed["hitter_sum"].fillna(0.0), keyed["hitter_n"].fillna(0.0)
    pair_sum, pair_n = keyed["pair_sum"].fillna(0.0), keyed["pair_n"].fillna(0.0)
    return ((hitter_sum - pair_sum + SHRINK_K * league) / (hitter_n - pair_n + SHRINK_K)).to_numpy()


def add_swing_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the per-swing columns listed in the module docstring. Needs is_bip,
    launch_speed, bat_speed, release_speed, batter, season, pitcher, stand, plate_x,
    miss_distance and the two intercept columns."""
    quality = contact_quality(df)
    has_contact_point = finite(df[DEPTH_COL]) & finite(df[REACH_COL])
    candidates = quality.notna() & has_contact_point
    rank = quality.where(candidates).groupby([df["batter"], df["season"]]).rank(pct=True)
    top = candidates & (rank > TOP_QUANTILE)

    df["ideal_depth"] = leave_pitcher_out_ideal(df, top, DEPTH_COL)
    df["ideal_reach"] = leave_pitcher_out_ideal(df, top, REACH_COL)

    df["timing_dev"] = df[DEPTH_COL] - df["ideal_depth"]
    df["timing_dev_abs"] = df["timing_dev"].abs()
    df["align_dev"] = df[REACH_COL] - df["ideal_reach"]
    df["align_dev_abs"] = df["align_dev"].abs()
    df["whiffmiss_log"] = np.log(df["miss_distance"].where(df["miss_distance"] > 0))
    side = df["stand"].astype(str).map({"R": -1.0, "L": 1.0})  # missing or unknown side -> NaN, not a guess
    df["plate_x_inside"] = df["plate_x"] * side
    return df
