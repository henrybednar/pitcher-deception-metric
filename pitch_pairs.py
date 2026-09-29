"""
Pitcher Deception Project: previous-pitch features
==================================================
Context available to the hitter before each pitch: how the previous pitch in the same at-bat
differs from this one. Built from the pitcher's own consecutive pitches, so no outcome enters
any feature.

  velo_diff_prev        release_speed - previous release_speed (mph)
  loc_diff_prev         distance between the two plate locations (ft)
  movement_diff_prev    distance between the two (ivb_in, hb_in) pairs (in)
  pitch_type_changed    1 if the pitch type changed, 0 if not, NaN on the first pitch
  spin_mirror_prev      circular spin-axis distance to the previous pitch, 0 for a match or a mirror
  tunnel_diff_prev      plate separation - separation at the 23.5 ft decision point (in). Positive
                        means the two pitches were closer at the decision point than at the plate.
  tunnel_x_changed, absvelo_x_changed, movement_x_changed
                        the last three multiplied by pitch_type_changed, so the effect of each can
                        differ between a repeat and a change of pitch

Measured on four-seamers and sliders, adding these to the timing model raised out-of-fold R2 by
about 0.003 to 0.006 and absorbed 8 to 9 percent of the pitcher-season timing residual. They
barely move whiff (AUC +0.001).
"""

import numpy as np
import pandas as pd

from driver_features import Y_TUNNEL, circular_mirror_score, traj_xy_at

SEQUENCE_KEYS = ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
AT_BAT = ["pitcher", "game_pk", "at_bat_number"]
TRAJECTORY_COLS = ["vx0", "vy0", "vz0", "ax", "ay", "az"]
PAIR_FEATURES = [
    "velo_diff_prev", "loc_diff_prev", "movement_diff_prev", "pitch_type_changed", "spin_mirror_prev",
    "tunnel_diff_prev", "tunnel_x_changed", "absvelo_x_changed", "movement_x_changed",
]
NEEDED = SEQUENCE_KEYS + TRAJECTORY_COLS + [
    "pitch_type", "release_speed", "plate_x", "plate_z", "ivb_in", "hb_in", "spin_axis"]


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise KeyError(f"add_pair_features needs columns {missing}")
    s = df.loc[df[SEQUENCE_KEYS].sort_values(SEQUENCE_KEYS).index, NEEDED].copy()
    x_t, z_t = traj_xy_at(s["plate_x"].values, s["plate_z"].values, s["vx0"].values, s["vy0"].values,
                          s["vz0"].values, s["ax"].values, s["ay"].values, s["az"].values, Y_TUNNEL)
    s["_xt"], s["_zt"] = x_t, z_t
    grp = s.groupby(AT_BAT, sort=False)
    prev = {c: grp[c].shift(1) for c in ["release_speed", "plate_x", "plate_z", "ivb_in", "hb_in", "spin_axis",
                                         "pitch_type", "_xt", "_zt"]}

    out = pd.DataFrame(index=s.index)
    out["velo_diff_prev"] = s["release_speed"] - prev["release_speed"]
    plate_dist = np.hypot(s["plate_x"] - prev["plate_x"], s["plate_z"] - prev["plate_z"])
    out["loc_diff_prev"] = plate_dist
    out["movement_diff_prev"] = np.hypot(s["ivb_in"] - prev["ivb_in"], s["hb_in"] - prev["hb_in"])
    changed = (s["pitch_type"] != prev["pitch_type"]).astype(float).where(prev["pitch_type"].notna())
    out["pitch_type_changed"] = changed
    out["spin_mirror_prev"] = circular_mirror_score(s["spin_axis"].values, prev["spin_axis"].values)
    tunnel_dist = np.hypot(s["_xt"] - prev["_xt"], s["_zt"] - prev["_zt"])
    out["tunnel_diff_prev"] = (plate_dist - tunnel_dist) * 12
    out["tunnel_x_changed"] = out["tunnel_diff_prev"] * changed
    out["absvelo_x_changed"] = out["velo_diff_prev"].abs() * changed
    out["movement_x_changed"] = out["movement_diff_prev"] * changed
    for col in PAIR_FEATURES:
        df[col] = out[col].reindex(df.index)
    return df
