import numpy as np
import pandas as pd
import pytest

import season_pairs as sp


def three_seasons():
    return pd.DataFrame({"pitcher": [1, 1, 1, 2, 2, 3, 3], "season": [2024, 2025, 2026, 2024, 2026, 2025, 2026],
                         "score": [90.0, 100.0, 110.0, 80.0, 120.0, 95.0, np.nan]})


def test_a_pitcher_with_three_seasons_gives_two_pairs_and_a_gap_year_gives_none():
    pairs = sp.consecutive_pairs(three_seasons(), ["score"]).sort_values(["pitcher", "season"])

    assert pairs[["pitcher", "season"]].values.tolist() == [[1, 2024], [1, 2025]]      # 2 skipped 2025, 3's 2026 is missing
    assert pairs["score_1"].tolist() == [90.0, 100.0]
    assert pairs["score_2"].tolist() == [100.0, 110.0]


def test_pooled_correlation_and_the_per_pair_correlations_agree_on_a_perfect_line():
    ps = pd.DataFrame({"pitcher": [1, 2, 3, 1, 2, 3, 1, 2, 3], "season": [2024] * 3 + [2025] * 3 + [2026] * 3,
                       "score": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 4.0, 8.0, 12.0]})
    pairs = sp.consecutive_pairs(ps, ["score"])

    assert sp.pooled_correlation(pairs, "score") == pytest.approx(1.0)
    assert sp.correlation_by_pair(pairs, "score") == pytest.approx({"2024-2025": 1.0, "2025-2026": 1.0})


def test_the_cluster_bootstrap_interval_brackets_the_correlation_and_is_wider_than_a_row_bootstrap_when_pairs_repeat():
    rng = np.random.default_rng(2)
    talent = rng.normal(0, 1, 120)
    pitchers = np.repeat(np.arange(120), 2)
    x = np.repeat(talent, 2) + rng.normal(0, 0.2, 240)
    y = np.repeat(talent, 2) * 0.5 + rng.normal(0, 1, 240)

    lo, hi = sp.cluster_bootstrap_interval(x, y, pitchers, n_boot=400)
    lo_rows, hi_rows = sp.cluster_bootstrap_interval(x, y, np.arange(240), n_boot=400)

    assert lo < np.corrcoef(x, y)[0, 1] < hi
    assert hi - lo > hi_rows - lo_rows


def test_a_repeated_pitcher_season_is_refused_instead_of_multiplying_pairs():
    ps = pd.DataFrame({"pitcher": [1, 1, 1], "season": [2024, 2024, 2025], "score": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        sp.consecutive_pairs(ps, ["score"])
