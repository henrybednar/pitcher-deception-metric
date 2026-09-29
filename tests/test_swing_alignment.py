import numpy as np
import pandas as pd
import pytest

import swing_alignment as sa

DEPTH, REACH = sa.DEPTH_COL, sa.REACH_COL


def swing_frame(rows: list[dict]) -> pd.DataFrame:
    base = {"batter": 1, "season": 2025, "pitcher": 10, "is_bip": True, "launch_speed": 90.0, "bat_speed": 70.0,
            "release_speed": 90.0, "stand": "R", "plate_x": 0.5, "miss_distance": np.nan,
            DEPTH: 30.0, REACH: 37.0}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_contact_quality_is_nan_off_balls_in_play_and_centered_on_them():
    rng = np.random.default_rng(0)
    n = 200
    frame = swing_frame([{}] * n)
    frame["bat_speed"] = rng.normal(70, 4, n)
    frame["release_speed"] = rng.normal(92, 3, n)
    frame["launch_speed"] = 4 + 0.93 * frame["bat_speed"] + 0.2 * frame["release_speed"] + rng.normal(0, 5, n)
    frame.loc[:9, "is_bip"] = False

    quality = sa.contact_quality(frame)

    assert quality.iloc[:10].isna().all()
    assert quality.iloc[10:].notna().all()
    assert abs(quality.iloc[10:].mean()) < 1e-9


def test_ideal_excludes_the_scored_pitchers_own_balls_and_shrinks_to_league():
    frame = swing_frame([
        {"pitcher": 1, DEPTH: 20.0}, {"pitcher": 1, DEPTH: 20.0},
        {"pitcher": 2, DEPTH: 40.0}, {"pitcher": 2, DEPTH: 40.0},
    ])
    top = pd.Series(True, index=frame.index)

    ideal = sa.leave_pitcher_out_ideal(frame, top, DEPTH)

    league = 30.0
    expected_for_pitcher_1 = (80.0 + sa.SHRINK_K * league) / (2 + sa.SHRINK_K)
    expected_for_pitcher_2 = (40.0 + sa.SHRINK_K * league) / (2 + sa.SHRINK_K)
    assert ideal[0] == pytest.approx(expected_for_pitcher_1)
    assert ideal[2] == pytest.approx(expected_for_pitcher_2)


def test_hitter_with_no_top_balls_falls_back_to_the_league_value():
    frame = swing_frame([{"batter": 1, DEPTH: 20.0}, {"batter": 2, DEPTH: 40.0}])
    top = pd.Series([True, False], index=frame.index)

    ideal = sa.leave_pitcher_out_ideal(frame, top, DEPTH)

    assert ideal[1] == pytest.approx(20.0)  # league mean of the only top ball


def test_add_swing_alignment_signs_and_absolute_values():
    rows = [{"pitcher": p, "launch_speed": 100.0 - i, DEPTH: 30.0, REACH: 37.0}
            for i, p in enumerate([1, 2, 3, 4, 5, 6, 7, 8])]
    rows.append({"pitcher": 9, "is_bip": False, "launch_speed": np.nan, DEPTH: 45.0, REACH: 30.0})
    rows.append({"pitcher": 9, "is_bip": False, "launch_speed": np.nan, DEPTH: 15.0, REACH: 44.0})
    frame = sa.add_swing_alignment(swing_frame(rows))

    early, late = frame.iloc[-2], frame.iloc[-1]
    assert early["timing_dev"] > 0 and late["timing_dev"] < 0
    assert early["timing_dev_abs"] == pytest.approx(abs(early["timing_dev"]))
    assert early["align_dev"] < 0 and late["align_dev"] > 0  # small reach = tied up, large = flail
    assert (frame["timing_dev_abs"] >= 0).all()


def populated(extra_rows: list[dict]) -> pd.DataFrame:
    """Eight balls in play with distinct exit velocities, so a top quartile exists, plus the rows under test."""
    base = [{"pitcher": p, "launch_speed": 100.0 - i} for i, p in enumerate(range(1, 9))]
    return swing_frame(base + extra_rows)


def test_missing_intercept_gives_missing_deviation():
    frame = sa.add_swing_alignment(populated([{"pitcher": 9, "is_bip": False, DEPTH: np.nan, REACH: np.nan}]))

    assert np.isnan(frame.iloc[-1]["timing_dev"]) and np.isnan(frame.iloc[-1]["align_dev_abs"])


def test_plate_x_inside_flips_by_batter_side_and_is_missing_when_side_is_unknown():
    frame = sa.add_swing_alignment(populated([
        {"pitcher": 9, "stand": "R", "plate_x": 0.4}, {"pitcher": 9, "stand": "L", "plate_x": 0.4},
        {"pitcher": 9, "stand": np.nan, "plate_x": 0.4},
    ]))

    inside = frame["plate_x_inside"].iloc[-3:].tolist()
    assert inside[:2] == [-0.4, 0.4]
    assert np.isnan(inside[2])


def test_whiffmiss_log_only_for_positive_distances():
    frame = sa.add_swing_alignment(populated([
        {"pitcher": 9, "miss_distance": 2.0}, {"pitcher": 9, "miss_distance": 0.0}, {"pitcher": 9},
    ]))

    tail = frame["whiffmiss_log"].iloc[-3:]
    assert tail.iloc[0] == pytest.approx(np.log(2.0))
    assert tail.iloc[1:].isna().all()


@pytest.mark.parametrize("column", ["launch_speed", "bat_speed", "release_speed", DEPTH])
def test_one_infinite_value_does_not_poison_the_other_rows(column):
    clean = sa.add_swing_alignment(populated([]))
    frame = populated([])
    frame.loc[3, column] = np.inf

    dirty = sa.add_swing_alignment(frame)

    others = [i for i in range(len(frame)) if i != 3]
    assert np.isfinite(dirty.loc[others, "ideal_depth"]).all()
    assert dirty.loc[others, "timing_dev_abs"].notna().all()
    assert np.isfinite(clean["ideal_depth"]).all()


def test_estimating_an_ideal_without_any_usable_ball_fails_loudly():
    frame = swing_frame([{"is_bip": False}, {"is_bip": False}])

    with pytest.raises(ValueError, match="no top-quality balls"):
        sa.add_swing_alignment(frame)
