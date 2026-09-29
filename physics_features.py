"""
Pitcher Deception Project: pitch-physics features
=================================================
Geometry of the Statcast trajectory that a hitter's eye sees late in the flight. All of it comes
from one pitch's own tracking (release speed and extension, initial velocity, constant acceleration),
so no outcome and no other pitch enters.

  late_break_x, late_break_z   inches of movement added after the 23.5 ft decision point, from the
                               acceleration alone (gravity removed from z)
  late_break_mag               size of that late movement
  decel_frac                   share of the initial plate-ward speed lost to drag by the plate
  ext_gain                     effective speed minus release speed (mph): what extension adds to
                               perceived velocity
  spin_axis_gap                circular hours between the measured spin axis and the axis implied by
                               observed movement (see below) — the one feature here that checks one
                               measurement against another rather than describing the trajectory alone

Measured on four-seamers with the shipped model, adding late_break/decel/ext_gain lowered whiff
logloss by 0.0005 (95% interval 0.0002 to 0.0008) and timing squared error by 0.11%, and left weak
contact unchanged.

tunnel_frac (share of flight time used by 23.5 ft) and ext_x_decel (extension times decel_frac) were
part of that same original bundle but never carried their own weight: permutation importance was
near-zero-or-negative across all 8 scored outcomes, tunnel_frac correlates 1.000 with decel_frac
(pure redundancy), and a held-out test confirmed dropping both changes nothing (whiff AUC delta
+0.0003, 95% interval -0.0003 to +0.0009; timing R2 delta -0.0000, interval -0.0003 to +0.0003).
Removed rather than kept as inert.

spin_axis_gap: the axis implied by movement is atan2(hb_in, -ivb_in), the sign convention that lines
up with Statcast's own spin_axis on four-seamers and curveballs, where spin efficiency is high enough
that movement should track the axis tightly (median gap 0.29h there, calibrated against 8
candidate sign/argument conventions). A real gap is seam-shifted wake: aerodynamic break a hitter's
eye doesn't expect from the spin it reads. Added for whiff AUC on four-seamers +0.0011 (95% interval
+0.0004 to +0.0018, all 5 folds positive); no effect on timing (interval includes zero).
"""

import numpy as np
import pandas as pd

from driver_features import Y0, Y_PLATE, Y_TUNNEL

GRAVITY = 32.174  # ft/s^2
INCHES_PER_FOOT = 12
PHYSICS_FEATURES = [
    "late_break_x", "late_break_z", "late_break_mag", "decel_frac", "ext_gain", "spin_axis_gap",
]


def time_from_release(vy0: pd.Series, ay: pd.Series, y: float) -> pd.Series:
    """Seconds from release until the ball is y feet from home plate, under constant acceleration."""
    vy = -np.sqrt(np.maximum(vy0 ** 2 - 2 * ay * (Y0 - y), 0))
    return (vy - vy0) / ay


def add_spin_axis_gap(df: pd.DataFrame) -> pd.Series:
    """Circular hours (0-6) between measured spin_axis and the axis implied by observed movement.
    NaN where spin_axis, hb_in or ivb_in is missing; HistGradientBoosting routes that natively, the
    same way it already does for the vy0/ay/ax/az inputs above, none of which is in REQUIRED_TRACKING
    either."""
    implied_deg = (np.degrees(np.arctan2(df["hb_in"], -df["ivb_in"])) + 360) % 360
    implied_clock = (implied_deg / 30.0) % 12
    measured_clock = (df["spin_axis"] / 30.0) % 12
    gap = (measured_clock - implied_clock + 6) % 12 - 6
    return gap.abs()


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds PHYSICS_FEATURES. Needs vy0, ay, ax, az, release_speed, effective_speed, release_extension,
    spin_axis, hb_in, ivb_in."""
    vy0, ay = df["vy0"], df["ay"]
    t_plate = time_from_release(vy0, ay, Y_PLATE)
    t_tunnel = time_from_release(vy0, ay, Y_TUNNEL)
    late = t_plate - t_tunnel
    df["late_break_x"] = 0.5 * df["ax"] * late ** 2 * INCHES_PER_FOOT
    df["late_break_z"] = 0.5 * (df["az"] + GRAVITY) * late ** 2 * INCHES_PER_FOOT
    df["late_break_mag"] = np.hypot(df["late_break_x"], df["late_break_z"])
    vy_plate = -np.sqrt(np.maximum(vy0 ** 2 - 2 * ay * (Y0 - Y_PLATE), 0))
    df["decel_frac"] = 1 - vy_plate.abs() / vy0.abs()
    df["ext_gain"] = df["effective_speed"] - df["release_speed"]
    df["spin_axis_gap"] = add_spin_axis_gap(df)
    return df
