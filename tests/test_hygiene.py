import numpy as np
import pandas as pd
import pytest

import pitch_hygiene as ph
from pitch_pairs import PAIR_FEATURES, add_pair_features


def test_nonfinite_values_become_nan():
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": ["x", "y", "z"]})

    out = ph.replace_nonfinite(df)

    assert out["a"].isna().tolist() == [False, True, True]


def test_domain_gates_clear_impossible_values_and_count_them():
    df = pd.DataFrame({"plate_z": [2.5, -57.6, 9.0], "release_extension": [6.3, 0.0, 6.0], "plate_x": [0.1, 35.0, -0.2]})

    out, cleared = ph.apply_domain_gates(df)

    assert out["plate_z"].isna().tolist() == [False, True, True]
    assert out["release_extension"].isna().tolist() == [False, True, False]
    assert out["plate_x"].isna().tolist() == [False, True, False]
    assert cleared["plate_z"] == 2 and cleared["release_extension"] == 1


def test_domain_gates_leave_ordinary_pitches_alone():
    df = pd.DataFrame({"plate_z": [0.5, 2.5, 4.4], "release_extension": [5.5, 6.5, 7.2], "release_speed": [70.0, 95.0, 101.0]})

    out, cleared = ph.apply_domain_gates(df.copy())

    assert out.equals(df)
    assert sum(cleared.values()) == 0


def test_pitch_type_aliases_and_blanking():
    df = pd.DataFrame({
        "pitcher": [1] * 4 + [2] * 3, "season": 2025,
        "pitch_type": ["SV", "FO", "KN", "FF", "FF", "SL", "CH"],
        "release_speed": [82.0, 84.0, 70.0, 95.0, 55.0, 52.0, 50.0],
    })

    out = ph.blank_unscored_pitch_types(ph.alias_pitch_types(df))

    assert out["pitch_type"].iloc[:2].tolist() == ["ST", "FS"]   # aliases
    assert pd.isna(out["pitch_type"].iloc[2])           # knuckleball is not scored
    assert out["pitch_type"].iloc[3] == "FF"
    assert out["pitch_type"].iloc[4:].isna().all()      # pitcher 2 averages 52 mph, a position player


def pair_frame() -> pd.DataFrame:
    base = dict(pitcher=1, game_pk=1, at_bat_number=1, vx0=-5.0, vy0=-135.0, vz0=-4.0, ax=-8.0, ay=25.0, az=-20.0, spin_axis=200.0,
                ivb_in=16.0, hb_in=8.0)
    rows = [
        dict(base, pitch_number=1, pitch_type="FF", release_speed=95.0, plate_x=0.0, plate_z=2.5),
        dict(base, pitch_number=2, pitch_type="FF", release_speed=94.0, plate_x=0.2, plate_z=2.4),
        dict(base, pitch_number=3, pitch_type="CH", release_speed=85.0, plate_x=0.5, plate_z=1.8, ivb_in=8.0, hb_in=14.0, spin_axis=110.0),
    ]
    return pd.DataFrame(rows).sample(frac=1, random_state=0)   # shuffled: the function must sort by pitch_number


def test_pair_features_use_the_previous_pitch_in_the_same_at_bat():
    out = add_pair_features(pair_frame()).sort_values("pitch_number")

    assert out["pitch_type_changed"].isna().iloc[0]                       # first pitch has no partner
    assert out["pitch_type_changed"].tolist()[1:] == [0.0, 1.0]
    assert out["velo_diff_prev"].tolist()[1:] == pytest.approx([-1.0, -9.0])
    assert out["movement_diff_prev"].iloc[2] == pytest.approx(np.hypot(8.0, 6.0))
    assert out["tunnel_x_changed"].iloc[1] == pytest.approx(0.0)          # repeat pitch: interaction is zero
    assert set(PAIR_FEATURES) <= set(out.columns)


def test_pair_features_do_not_cross_at_bats():
    frame = pair_frame()
    frame.loc[frame["pitch_number"] == 3, "at_bat_number"] = 2

    out = add_pair_features(frame).sort_values("pitch_number")

    assert out["velo_diff_prev"].iloc[2] != out["velo_diff_prev"].iloc[2]  # NaN: new at-bat, no partner
