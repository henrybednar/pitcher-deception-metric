import numpy as np
import pandas as pd
import pytest

import out_of_time as oot


def test_shift_seasons_moves_every_row_without_touching_the_rest():
    frame = pd.DataFrame({"season": [2024, 2025], "stuff_plus": [101.0, 99.0]})

    out = oot.shift_seasons(frame)

    assert out["season"].tolist() == [2025, 2026]
    assert out["stuff_plus"].tolist() == [101.0, 99.0] and frame["season"].tolist() == [2024, 2025]


def test_holdout_rows_keeps_one_real_season_labelled_as_another():
    chunks = [pd.DataFrame({"season": [2025, 2026], "x": [1, 2]}), pd.DataFrame({"season": [2026], "x": [3]}),
              pd.DataFrame({"season": [2025], "x": [4]})]

    out = pd.concat(list(oot.holdout_rows(chunks, 2025, 2026)), ignore_index=True)

    assert out["x"].tolist() == [1, 4] and (out["season"] == 2026).all()


def season_pair(n=300, seed=0):
    rng = np.random.default_rng(seed)
    talent = rng.normal(0, 1, n)
    rows = []
    for season in (2025, 2026):
        row = pd.DataFrame({"pitcher": np.arange(n), "season": season, "qualified": True})
        for label in ("a", "b"):
            row[f"{label}_index"] = 100 + 10 * (talent + rng.normal(0, 1, n))
            row[f"{label}_n"] = 100.0
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def test_weighted_composite_is_scaled_to_100_and_10_on_the_qualified_pitchers():
    ps = season_pair()

    score = oot.weighted_composite(ps, ["a", "b"], {"a": 0.5, "b": 0.5})

    assert score.mean() == pytest.approx(100.0, abs=1e-6) and score.std() == pytest.approx(10.0, abs=1e-6)


def test_weighted_composite_gives_no_weight_to_an_unreliable_member_and_needs_two_members():
    ps = season_pair()

    one_usable = oot.weighted_composite(ps, ["a", "b"], {"a": 0.5, "b": 0.05})            # b is under the 0.2 floor, so only a counts

    assert one_usable.isna().all()


def test_qualified_pair_yoy_reflects_shared_talent_and_brackets_with_an_interval():
    ps = season_pair()

    r, lo, hi, n = oot.qualified_pair_yoy(ps, oot.weighted_composite(ps, ["a", "b"], {"a": 0.5, "b": 0.5}))

    assert n == 300 and lo < r < hi and 0.3 < r < 0.8
