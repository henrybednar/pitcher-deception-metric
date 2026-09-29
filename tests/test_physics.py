import numpy as np
import pandas as pd
import pytest

from physics_features import GRAVITY, PHYSICS_FEATURES, add_physics_features, time_from_release


def pitch(**overrides):
    row = dict(vy0=-130.0, ay=25.0, ax=0.0, az=-GRAVITY, release_speed=92.0, effective_speed=94.0, release_extension=6.5,
              hb_in=-8.0, ivb_in=16.0, spin_axis=225.0)
    row.update(overrides)
    return pd.DataFrame([row])


def test_flight_time_to_the_plate_is_about_four_tenths_of_a_second_for_a_92_mph_pitch():
    t = time_from_release(pd.Series([-130.0]), pd.Series([25.0]), 17 / 12)

    assert t.iloc[0] == pytest.approx(0.40, abs=0.02)


def test_a_pitch_with_no_sideways_or_extra_vertical_acceleration_has_no_late_break():
    out = add_physics_features(pitch())

    assert out.loc[0, "late_break_x"] == pytest.approx(0.0)
    assert out.loc[0, "late_break_z"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc[0, "late_break_mag"] == pytest.approx(0.0, abs=1e-9)


def test_late_break_grows_with_acceleration_and_has_the_right_sign():
    small = add_physics_features(pitch(ax=5.0)).loc[0, "late_break_x"]
    large = add_physics_features(pitch(ax=-20.0)).loc[0, "late_break_x"]

    assert small > 0 > large
    assert abs(large) == pytest.approx(4 * small)


def test_extension_gain_is_effective_speed_minus_release_speed():
    assert add_physics_features(pitch()).loc[0, "ext_gain"] == pytest.approx(2.0)


def test_deceleration_fraction_is_a_share_between_zero_and_one():
    out = add_physics_features(pitch())

    assert 0 < out.loc[0, "decel_frac"] < 0.2


def test_every_physics_feature_is_added_and_finite_for_a_normal_pitch():
    out = add_physics_features(pitch(ax=8.0, az=-20.0))

    assert set(PHYSICS_FEATURES) <= set(out.columns)
    assert np.isfinite(out[PHYSICS_FEATURES].to_numpy()).all()


def implied_axis_deg(hb: float, ivb: float) -> float:
    return (np.degrees(np.arctan2(hb, -ivb)) + 360) % 360


def test_spin_axis_gap_is_zero_when_measured_axis_matches_the_movement_implied_axis():
    hb, ivb = -8.0, 16.0
    out = add_physics_features(pitch(hb_in=hb, ivb_in=ivb, spin_axis=implied_axis_deg(hb, ivb)))

    assert out.loc[0, "spin_axis_gap"] == pytest.approx(0.0, abs=1e-9)


def test_spin_axis_gap_is_six_hours_when_the_axes_are_opposite():
    hb, ivb = -8.0, 16.0
    opposite = (implied_axis_deg(hb, ivb) + 180) % 360
    out = add_physics_features(pitch(hb_in=hb, ivb_in=ivb, spin_axis=opposite))

    assert out.loc[0, "spin_axis_gap"] == pytest.approx(6.0, abs=1e-9)


def test_spin_axis_gap_is_nan_when_spin_axis_is_missing():
    out = add_physics_features(pitch(spin_axis=np.nan))

    assert np.isnan(out.loc[0, "spin_axis_gap"])
