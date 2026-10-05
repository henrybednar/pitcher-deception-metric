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


def stuff_cells(types=(("SL", 100.0, 0.0), ("CH", 60.0, 1.5)), n=120, slope=0.05, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for pitch_type, centre, shift in types:
        stuff = rng.normal(centre, 8, n)
        parts.append(pd.DataFrame({"pitcher": np.arange(n) + len(parts) * n, "pitch_type": pitch_type, "n": 100,
                                   "stuff": stuff, "resid": shift + slope * (stuff - centre) + rng.normal(0, 0.2, n)}))
    return pd.concat(parts, ignore_index=True)


def test_stuff_slope_recovers_the_residual_change_per_ten_stuff_points():
    cells = stuff_cells(types=(("SL", 100.0, 0.0),))

    result = sr.stuff_slope(cells)

    assert result["per_10"] == pytest.approx(0.5, abs=0.1)
    assert result["lo"] < result["per_10"] < result["hi"]
    assert result["cells"] == 120 and result["pitchers"] == 120


def test_stuff_slope_standard_error_grows_when_a_pitchers_samples_are_clustered():
    cells = stuff_cells(types=(("SL", 100.0, 0.0),))
    doubled = pd.concat([cells, cells], ignore_index=True)           # each pitcher's second sample repeats the first

    independent = sr.stuff_slope(doubled.assign(pitcher=np.arange(len(doubled))))
    clustered = sr.stuff_slope(doubled)

    assert clustered["se"] > independent["se"] * 1.2


def test_slopes_by_pitch_type_reports_each_type_and_a_pooled_slope_within_type():
    cells = stuff_cells()                                            # changeups sit lower on Stuff+ and higher on residual

    result = sr.slopes_by_pitch_type(cells)

    assert set(result) == {"SL", "CH", "all"}
    assert result["SL"]["per_10"] == pytest.approx(0.5, abs=0.15)
    assert result["CH"]["per_10"] == pytest.approx(0.5, abs=0.15)
    assert result["all"]["per_10"] == pytest.approx(0.5, abs=0.1)    # a pooled fit without type effects would read negative


def test_slopes_by_pitch_type_skips_a_type_with_too_few_samples():
    cells = pd.concat([stuff_cells(types=(("SL", 100.0, 0.0),)), stuff_cells(types=(("CH", 60.0, 0.0),), n=5)], ignore_index=True)

    assert "CH" not in sr.slopes_by_pitch_type(cells)


def test_residual_cells_can_be_restricted_to_a_subset_of_pitches_with_a_lower_minimum():
    df = pd.DataFrame({"pitcher": 1, "season": 2025, "pitch_type": "FF", "is_swing": True,
                       "is_whiff": [1, 0] * 30, "whiff_expected_full": 0.40})
    first = pd.Series([i < 30 for i in range(60)])                   # 30 of the 60 swings

    assert sr.residual_cells(df, "whiff", restrict=first).empty                       # 30 is under the usual 50
    cells = sr.residual_cells(df, "whiff", restrict=first, min_n_scale=0.5)
    assert cells.loc[0, "n"] == 30 and cells.loc[0, "resid"] == pytest.approx((0.5 - 0.4) * 100)


def test_first_pitch_flags_mark_pitch_number_one_after_dropping_non_regular_season_rows(tmp_path):
    raw = pd.DataFrame({"pitcher": [1, 1, 1, 2], "season": 2025, "game_type": ["S", "R", "R", "R"], "pitch_number": [1, 1, 2, 1]})
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    scored = pd.DataFrame({"pitcher": [1, 1, 2], "season": 2025})

    flags = sr.first_pitch_flags(str(path), scored)

    assert flags.tolist() == [True, False, True]


def test_first_pitch_flags_refuse_rows_that_do_not_line_up(tmp_path):
    raw = pd.DataFrame({"pitcher": [1, 2], "season": 2025, "game_type": "R", "pitch_number": [1, 1]})
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)

    with pytest.raises(ValueError):
        sr.first_pitch_flags(str(path), pd.DataFrame({"pitcher": [2, 1], "season": 2025}))


def test_stuff_plus_member_correlations_are_each_members_own_index_against_stuff_plus_not_against_deception_plus():
    stuff = np.arange(60, dtype=float)
    ps = pd.DataFrame({"qualified": True, "season": [2025, 2026] * 30, "stuff_plus": stuff,
                       "deception_plus": np.random.default_rng(0).normal(100, 10, 60)})
    for member, sign in (("whiff", 1.0), ("chase", -1.0), ("weak", 1.0), ("timing", -1.0), ("whiffmiss", 1.0)):
        ps[f"{member}_index"] = sign * stuff

    members = stuff_plus_correlations(ps)["members"]

    assert members == pytest.approx({"whiff": 1.0, "chase": -1.0, "weak": 1.0, "timing": -1.0, "whiffmiss": 1.0})
