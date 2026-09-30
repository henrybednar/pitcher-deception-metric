"""
Pitcher Deception Project: per-pitch table
==========================================
Reads the pitch-level Statcast file, builds every feature and target once (load_pitch_data), and
writes per_pitch_predictions.csv with the identifying columns, the outcome flags and the half
assignment. The fit_* steps that follow each append their out-of-fold predictions to that file by
position, after read_aligned_predictions() checks that the row order matches.

The swing-timing outcomes (contact depth, horizontal alignment, whiff miss distance) get their
targets here through swing_alignment.add_swing_alignment(), because they need each hitter's ideal
contact point, not just the raw intercept columns.

HALF-SPLIT: every pitch also gets a deterministic half assignment (game_pk parity) for split-half
reliability. It is independent of chronology, so it measures measurement noise rather than
true-talent drift across the season.

REGULAR SEASON ONLY: postseason and spring training pitches are dropped (filter_regular_season).
They come from a different competitive population (playoff-caliber pitching; non-competitive spring
rosters) — postseason whiff rate runs 2 to 4 points above regular season, spring-training chase rate
9 points above — and the two seasons in this pull carry very different shares of them (2025 was 5.7%
non-regular-season, 2026 0.3%, since this pull's cutoff of September 27 predates the 2026
postseason). Left in, that mix would land unevenly on whichever pitchers happened to appear in
those games.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pitch_hygiene import (alias_pitch_types, apply_domain_gates, blank_incomplete_tracking, blank_unscored_pitch_types,
                           replace_nonfinite)
from physics_features import add_physics_features
from pitch_pairs import TRAJECTORY_COLS, add_pair_features
from swing_alignment import add_swing_alignment

PITCH_LEVEL_FILE = "raw/statcast_pitch_level_2025_2026.csv"
PREDICTIONS_FILE = "output/per_pitch_predictions.csv"
CHUNKSIZE = 500_000
WEAK_CONTACT_EV = 85.0
IN_ZONE = set(range(1, 10))
MIN_ZONE_HEIGHT = 0.5  # ft; real strike zones run about 1.1 to 2.6
ALIGN_COLS = ["pitcher", "pitch_type", "season", "game_pk", "half"]
TABLE_COLS = ALIGN_COLS + ["is_swing", "is_in_zone", "is_bip", "is_whiff", "is_gb", "is_weak"]

RAW_COLS = [
    "pitcher", "pitch_type", "season", "description", "zone", "p_throws",
    "game_pk", "bb_type", "launch_speed", "bat_speed", "miss_distance",
    "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "release_speed", "release_spin_rate", "ivb_in", "hb_in",
    "vaa", "haa", "release_extension", "effective_speed",
    "arm_angle", "release_pos_x", "release_pos_z",
    "batter", "stand", "fielder_2", "at_bat_number", "pitch_number", "spin_axis",
    "balls", "strikes", "home_team", "game_type",
    "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score",
] + TRAJECTORY_COLS

SWING_DESC = {"foul", "foul_tip", "hit_into_play", "swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_bunt"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


def normalize_plate_z(df: pd.DataFrame) -> pd.Series:
    """Plate height as a fraction of the batter's zone. NaN when the zone height is missing, zero or negative."""
    height = df["sz_top"] - df["sz_bot"]
    return (df["plate_z"] - df["sz_bot"]) / height.where(height >= MIN_ZONE_HEIGHT)


def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """Regular-season pitches only. game_type is dropped afterward — nothing downstream needs it,
    and its only job is this one filter."""
    return df[df["game_type"] == "R"].drop(columns="game_type").reset_index(drop=True)


def load_pitch_data(path: str) -> pd.DataFrame:
    """Read the pitch-level file and build every feature and target.

    Order matters: the regular-season filter and non-finite/impossible-tracking cleanup run first, so
    nothing downstream (least-squares fits, group means) can be poisoned or skewed by a game that
    shouldn't count. Pitch types are aliased before the previous-pitch features and blanked afterwards,
    so a scored pitch still sees the pitch before it.
    """
    df = pd.concat(pd.read_csv(path, usecols=RAW_COLS, chunksize=CHUNKSIZE, low_memory=False), ignore_index=True)
    df = filter_regular_season(df)
    df = replace_nonfinite(df)
    df, _ = apply_domain_gates(df)
    df = alias_pitch_types(df)

    df["is_swing"] = df["description"].isin(SWING_DESC)
    df["is_in_zone"] = df["zone"].isin(IN_ZONE)
    # A ball in play counts only when Statcast measured it (bb_type and launch_speed both present).
    # About 1% of balls in play have no exit velocity; scoring them as "not weak" biased weak contact.
    df["is_bip"] = (df["description"] == "hit_into_play") & df["launch_speed"].notna() & df["bb_type"].notna()
    df["is_whiff"] = df["description"].isin(WHIFF_DESC).astype(int)
    df["is_gb"] = (df["bb_type"] == "ground_ball").astype(int)
    df["is_weak"] = (df["launch_speed"] < WEAK_CONTACT_EV).astype(int)

    hand_sign = np.where(df["p_throws"] == "R", 1, -1)
    df["arm_side_break_in"] = df["hb_in"] * hand_sign
    df["release_pos_x_armside"] = df["release_pos_x"] * hand_sign
    df["plate_x_armside"] = df["plate_x"] * hand_sign
    df["plate_z_norm"] = normalize_plate_z(df)
    df["p_throws"] = df["p_throws"].astype("category")
    df = add_swing_alignment(df)
    df = add_pair_features(df)
    df = add_physics_features(df)
    df = blank_unscored_pitch_types(df)
    df = blank_incomplete_tracking(df)

    # Deterministic 50/50 split by game, independent of chronology, isolates
    # measurement noise from true-talent drift across the season.
    df["half"] = (df["game_pk"] % 2).astype(int)
    df = add_pitch_count_in_appearance(df)
    df = add_times_faced_this_game(df)
    df = add_game_state_features(df)

    return replace_nonfinite(df)


def add_pitch_count_in_appearance(df: pd.DataFrame) -> pd.DataFrame:
    """In-game fatigue: this pitch's count within the outing (1 = first pitch thrown). A held-out
    test found this explains most of the reliever-vs-starter whiff gap (about 1.3 points actual-
    minus-expected, shrunk to about 0.1 once the model can see it). Relievers are almost always on
    a fresh arm, starters routinely aren't, and the model had no way to tell them apart."""
    order = df.sort_values(["pitcher", "game_pk", "at_bat_number", "pitch_number"]).index
    df.loc[order, "pitch_count_in_appearance"] = df.loc[order].groupby(["pitcher", "game_pk"]).cumcount() + 1
    return df


def add_times_faced_this_game(df: pd.DataFrame) -> pd.DataFrame:
    """This at-bat's rank among the same pitcher-batter pair's meetings this game (1 = first time
    facing each other today). The pitch-level version of "times through the order": a batter's
    second or third look at the same pitcher in one game is a real, plausible advantage that the
    model had no way to see."""
    order = df.sort_values(["pitcher", "batter", "game_pk", "at_bat_number", "pitch_number"]).index
    keys = df.loc[order, ["pitcher", "batter", "game_pk", "at_bat_number"]]
    first_pitch_of_at_bat = ~keys.duplicated()
    rank = first_pitch_of_at_bat.groupby([keys["pitcher"], keys["batter"], keys["game_pk"]]).cumsum()
    df.loc[order, "times_faced_this_game"] = rank.to_numpy()
    return df


def add_game_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Situational context no other feature carries: outs, total baserunners, and the pitching
    team's score lead (positive) or deficit (negative). A batter's approach plausibly shifts with
    the bases empty in a blowout versus runners in scoring position in a close game, and the model
    had no way to see that either."""
    df["outs"] = df["outs_when_up"]
    df["runners_on"] = df[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1)
    df["score_diff"] = df["fld_score"] - df["bat_score"]
    return df


def read_aligned_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Read per_pitch_predictions.csv and refuse to go on unless its rows line up with df by position."""
    existing = pd.read_csv(PREDICTIONS_FILE)
    assert len(existing) == len(df), f"row count mismatch: {len(existing):,} vs {len(df):,}"
    left, right = existing[ALIGN_COLS].reset_index(drop=True), df[ALIGN_COLS].reset_index(drop=True)
    differs = pd.concat([(left[c] != right[c]) & ~(left[c].isna() & right[c].isna()) for c in ALIGN_COLS], axis=1)
    n_mismatch = int(differs.any(axis=1).sum())
    assert n_mismatch == 0, f"{n_mismatch:,} rows misaligned, refusing to merge by position"
    print("row alignment verified, safe to merge by position.", flush=True)
    return existing


def save_predictions(existing: pd.DataFrame, df: pd.DataFrame, new_cols: list[str]) -> None:
    """Append df's new_cols to the predictions file by position."""
    for c in new_cols:
        existing[c] = df[c].values
    existing.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nSaved {PREDICTIONS_FILE}: {existing.shape[0]:,} rows, {existing.shape[1]} cols. Done.", flush=True)


if __name__ == "__main__":
    print("loading full pitch-level data...", flush=True)
    df = load_pitch_data(PITCH_LEVEL_FILE)
    print(f"total pitches: {len(df):,}", flush=True)
    Path(PREDICTIONS_FILE).parent.mkdir(exist_ok=True)
    df[TABLE_COLS].to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved {PREDICTIONS_FILE}: {len(df):,} rows, {len(TABLE_COLS)} cols. Done.", flush=True)
