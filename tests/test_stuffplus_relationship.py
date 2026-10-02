import numpy as np
import pandas as pd
import pytest

import stuffplus_relationship as sr
from export_site_stats import stuff_plus_correlations


def test_cluster_weighted_mean_matches_the_plain_weighted_mean_and_a_hand_computed_standard_error():
    mean, se = sr.cluster_weighted_mean([1.0, 3.0], [1.0, 1.0], ["a", "b"])

    assert mean == pytest.approx(2.0)
    assert se == pytest.approx(np.sqrt(2) / 2)   # two clusters, scores of -1 and +1, divided by total weight 2


def test_cluster_weighted_mean_has_no_spread_when_every_sample_belongs_to_one_pitcher():
    # a single cluster's deviations cancel by construction, so it carries no information about spread
    _, se = sr.cluster_weighted_mean([1.0, 3.0], [1.0, 1.0], ["a", "a"])

    assert se == pytest.approx(0.0)


def test_fifths_are_ranked_within_each_pitch_type_not_across_them():
    cells = pd.DataFrame({
        "pitch_type": ["SL"] * 10 + ["CH"] * 5,
        "stuff": list(range(100, 110)) + list(range(60, 65)),  # changeups all rate far below sliders
    })

    fifths = sr.fifths_within_pitch_type(cells)

    assert fifths[:10].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert fifths[10:].tolist() == [1, 2, 3, 4, 5]


def test_fifths_table_follows_a_residual_that_rises_with_stuff():
    n = 100
    cells = pd.DataFrame({
        "pitcher": np.arange(n), "pitch_type": "SL", "n": 60,
        "stuff": np.linspace(80, 120, n), "resid": np.linspace(-2.0, 2.0, n),
    })

    rows = sr.fifths_table(cells)

    assert [r["fifth"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["cells"] for r in rows] == [20] * 5
    resid = [r["resid"] for r in rows]
    assert resid == sorted(resid) and resid[0] < 0 < resid[-1]
    assert all(r["se"] > 0 for r in rows)


def test_residual_cells_scales_binary_outcomes_to_points_and_drops_thin_samples():
    big = pd.DataFrame({"pitcher": 1, "season": 2025, "pitch_type": "FF", "is_swing": True,
                        "is_whiff": [1] * 20 + [0] * 40, "whiff_expected_full": 0.30})
    thin = pd.DataFrame({"pitcher": 2, "season": 2025, "pitch_type": "FF", "is_swing": True,
                         "is_whiff": [1] * 3 + [0] * 7, "whiff_expected_full": 0.30})

    cells = sr.residual_cells(pd.concat([big, thin], ignore_index=True), "whiff")

    assert cells["pitcher"].tolist() == [1]                       # 10 swings is below the 50 minimum
    assert cells.loc[0, "resid"] == pytest.approx((20 / 60 - 0.30) * 100)


def test_residual_cells_ignores_pitches_the_model_never_scored():
    scored = pd.DataFrame({"pitcher": 1, "season": 2025, "pitch_type": "FF", "is_swing": True,
                           "is_whiff": [1] * 30 + [0] * 30, "whiff_expected_full": 0.50})
    unscored = pd.DataFrame({"pitcher": [1], "season": [2025], "pitch_type": [None], "is_swing": [True],
                             "is_whiff": [1], "whiff_expected_full": [np.nan]})

    cells = sr.residual_cells(pd.concat([scored, unscored], ignore_index=True), "whiff")

    assert cells.loc[0, "n"] == 60
    assert cells.loc[0, "resid"] == pytest.approx(0.0)


def test_stuff_by_pitch_type_folds_forkballs_into_split_fingers_and_drops_missing_values():
    table = pd.DataFrame({
        "pitcher": [1, 1, 2], "season": 2025, "pitch_type": ["FO", "FF", "SL"],
        "stuff_plus_by_pitch_type": [95.0, 110.0, np.nan],
    })

    stuff = sr.stuff_by_pitch_type(table)

    assert sorted(zip(stuff["pitch_type"], stuff["stuff"])) == [("FF", 110.0), ("FS", 95.0)]


def test_stuff_plus_correlations_use_qualified_pitcher_seasons_with_a_stuff_figure_only():
    stuff = np.arange(60, dtype=float)
    ps = pd.DataFrame({
        "qualified": [True] * 50 + [False] * 10,
        "season": [2025, 2026] * 30,
        "stuff_plus": stuff,
        "deception_plus": 2.0 * stuff,
    })
    for member in ("whiff", "chase", "weak", "timing", "whiffmiss"):
        ps[f"{member}_index"] = stuff
    ps.loc[55, "stuff_plus"] = np.nan                      # unqualified anyway, and no Stuff+
    ps.loc[3, "stuff_plus"] = np.nan                       # a qualified pitcher-season with no Stuff+ figure is skipped
    ps.loc[55:59, "deception_plus"] = -stuff[55:60]        # unqualified rows must not move the answer

    result = stuff_plus_correlations(ps)

    assert result["n"] == 49
    assert result["composite"] == pytest.approx(1.0)
    assert result["by_season"] == pytest.approx({2025: 1.0, 2026: 1.0})
    assert set(result["members"]) == {"whiff", "chase", "weak", "timing", "whiffmiss"}
