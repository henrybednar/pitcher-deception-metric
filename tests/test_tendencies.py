import numpy as np
import pandas as pd
import pytest

from tendencies import build_totals, tendency_for_rows


def rows() -> pd.DataFrame:
    return pd.DataFrame({
        "batter": [1, 1, 1, 1, 1, 1], "season": [2025] * 6,
        "pitcher": [10, 10, 20, 20, 30, 40], "hit": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })


def totals_for(df: pd.DataFrame) -> pd.DataFrame:
    return build_totals(df, pd.Series(True, index=df.index), "batter", "hit")


def test_training_rows_leave_out_every_pitch_from_their_own_pitcher():
    df = rows()

    out = tendency_for_rows(df, totals_for(df), "batter", np.array([10, 20, 30, 40]), leave_out_own=True)

    # pitcher 10 sees pitchers 20, 30, 40: (0+1+0+1)/4; pitcher 20 sees 10, 30, 40: (1+1+0+1)/4
    assert out[:2].tolist() == pytest.approx([0.5, 0.5])
    assert out[2:4].tolist() == pytest.approx([0.75, 0.75])
    assert out[4] == pytest.approx(4 / 5)  # pitcher 30 sees 10, 20, 40: (1+1+0+1+1)/5
    assert out[5] == pytest.approx(3 / 5)  # pitcher 40 sees 10, 20, 30: (1+1+0+1+0)/5


def test_test_rows_use_only_training_pitchers():
    df = rows()
    train = np.array([10, 20])

    out = tendency_for_rows(df[df.pitcher.isin([30, 40])], totals_for(df), "batter", train, leave_out_own=False)

    assert out.tolist() == pytest.approx([3 / 4, 3 / 4])   # (1+1+0+1) over the four training pitches


def test_held_out_outcomes_cannot_change_a_test_rows_feature():
    df = rows()
    train = np.array([10, 20])
    test = df[df.pitcher.isin([30, 40])]
    before = tendency_for_rows(test, totals_for(df), "batter", train, leave_out_own=False)

    changed = df.copy()
    changed.loc[changed.pitcher.isin([30, 40]), "hit"] = [1.0, 1.0]
    after = tendency_for_rows(test, totals_for(changed), "batter", train, leave_out_own=False)

    assert after.tolist() == before.tolist()


def test_tendency_is_nan_when_no_other_training_pitcher_saw_the_hitter():
    df = pd.DataFrame({"batter": [1, 1], "season": [2025, 2025], "pitcher": [10, 10], "hit": [1.0, 0.0]})

    out = tendency_for_rows(df, totals_for(df), "batter", np.array([10]), leave_out_own=True)

    assert np.isnan(out).all()


def test_seasons_are_kept_apart():
    df = pd.DataFrame({"batter": [1, 1, 1], "season": [2025, 2025, 2026], "pitcher": [10, 20, 20], "hit": [1.0, 0.0, 1.0]})

    out = tendency_for_rows(df, totals_for(df), "batter", np.array([10, 20]), leave_out_own=True)

    assert out[0] == pytest.approx(0.0)           # 2025, pitcher 10 sees only pitcher 20's 2025 outcome
    assert np.isnan(out[2])                       # 2026 has a single pitcher


def test_split_by_hand_matches_a_row_against_same_handed_training_pitchers_only():
    df = pd.DataFrame({
        "batter": [1, 1, 1, 1, 1, 1], "season": [2025] * 6,
        "pitcher": [10, 10, 20, 20, 30, 40],
        "p_throws": ["R", "R", "L", "L", "R", "L"],
        "hit": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })
    train = np.array([10, 20, 30, 40])
    totals = build_totals(df, pd.Series(True, index=df.index), "batter", "hit", split_by_hand=True)

    out = tendency_for_rows(df, totals, "batter", train, leave_out_own=True, split_by_hand=True)

    assert out[0] == pytest.approx(0.0)    # pitcher 10 (R) sees only the other R pitcher, 30: 0/1
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(1.0)    # pitcher 20 (L) sees only the other L pitcher, 40: 1/1
    assert out[3] == pytest.approx(1.0)
    assert out[4] == pytest.approx(1.0)    # pitcher 30 (R) sees only the other R pitcher, 10: 2/2
    assert out[5] == pytest.approx(0.5)    # pitcher 40 (L) sees only the other L pitcher, 20: 1/2
