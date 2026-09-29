"""
Pitcher Deception Project: input hygiene for the pitch-level file
=================================================================
Applied once, inside load_pitch_data(), before any feature or target is built.

1. Non-finite values become NaN. One inf in a fitted column (a least-squares input, a group
   mean) would otherwise poison every row that shares it.
2. Physically impossible tracking values become NaN (DOMAIN_GATES). On the 2025-26 pull the
   gates touch about 140 of 1.4 million rows, for example plate_z = -57.6 or plate_x = 35 ft.
3. Pitch types are normalized in three steps (alias_pitch_types, then blank_unscored_pitch_types). SV (slurve) joins ST and FO (forkball) joins FS. Types that do
   not share one physical model (knuckleball, eephus, generic fastball, slow curve, screwball,
   unknown) and every pitch from a pitcher-season averaging under 75 mph (position players and
   other non-pitchers) get pitch_type NaN. NaN rows are never modeled or scored.
   A pitch missing any core tracking value (REQUIRED_TRACKING) also gets pitch_type NaN. Missing
   tracking is informative (swings with no effective_speed whiff 10% of the time against 23% for
   the rest), and a tree model routes a NaN it rarely saw in training to an arbitrary branch, so
   those pitches are left out of scoring instead of guessed at. Arm angle is exempt: it is missing
   on 2% of pitches, which is enough for the model to learn from.

Rows whose pitch_type is NaN keep their other columns, so features that describe the whole
data set (sequencing partners, tendencies) can still see them.
"""

import numpy as np
import pandas as pd

DOMAIN_GATES = {
    "release_speed": (25.0, 110.0),
    "effective_speed": (25.0, 115.0),
    "release_extension": (3.0, 9.0),
    "release_spin_rate": (0.0, 4500.0),
    "release_pos_x": (-6.0, 6.0),
    "release_pos_z": (0.0, 9.0),
    "sz_top": (2.0, 5.0),
    "sz_bot": (0.5, 2.5),
    "plate_x": (-5.0, 5.0),
    "plate_z": (-2.0, 8.0),
    "launch_speed": (0.0, 125.0),
    "bat_speed": (0.0, 100.0),
    "intercept_ball_minus_batter_pos_x_inches": (0.0, 80.0),
    "intercept_ball_minus_batter_pos_y_inches": (-30.0, 90.0),
}
REQUIRED_TRACKING = [
    "release_speed", "release_spin_rate", "ivb_in", "arm_side_break_in", "vaa", "haa", "release_extension",
    "effective_speed", "release_pos_z", "release_pos_x_armside", "plate_x_armside", "plate_z_norm",
]
PITCH_TYPE_ALIASES = {"SV": "ST", "FO": "FS"}
SCORED_PITCH_TYPES = frozenset({"FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS"})
MIN_PITCHER_MEAN_SPEED = 75.0


def replace_nonfinite(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    return df


def apply_domain_gates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set values outside DOMAIN_GATES to NaN. Returns the frame and how many values each gate cleared."""
    cleared = {}
    for col, (lo, hi) in DOMAIN_GATES.items():
        if col not in df.columns:
            continue
        bad = (df[col] < lo) | (df[col] > hi)
        cleared[col] = int(bad.sum())
        df[col] = df[col].mask(bad)
    return df, cleared


def alias_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    df["pitch_type"] = df["pitch_type"].replace(PITCH_TYPE_ALIASES)
    return df


def blank_unscored_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    """pitch_type -> NaN for types outside SCORED_PITCH_TYPES and for pitcher-seasons averaging under
    MIN_PITCHER_MEAN_SPEED. Run after any feature that needs the full pitch sequence."""
    slow = df.groupby(["pitcher", "season"])["release_speed"].transform("mean") < MIN_PITCHER_MEAN_SPEED
    unscored = ~df["pitch_type"].isin(SCORED_PITCH_TYPES) | slow
    df["pitch_type"] = df["pitch_type"].mask(unscored)
    return df


def blank_incomplete_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """pitch_type -> NaN for pitches missing any REQUIRED_TRACKING value. Run after those columns exist."""
    df["pitch_type"] = df["pitch_type"].mask(df[REQUIRED_TRACKING].isna().any(axis=1))
    return df
