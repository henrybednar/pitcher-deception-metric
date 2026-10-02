import numpy as np
import pandas as pd
import pytest

from export_site_stats import sequencing_feature_overlap


def test_sequencing_feature_overlap_reports_the_correlation_between_the_features_and_with_the_whiff_residual():
    rng = np.random.default_rng(0)
    repeat = rng.uniform(0.2, 0.6, 400)
    gap = 8 * (1 - repeat) + rng.normal(scale=0.2, size=400)          # more repeats, smaller average gap
    residual = 0.01 * gap + 0.02 * repeat + rng.normal(scale=0.05, size=400)
    pairs = pd.DataFrame({"pitcher": np.arange(400), "season": 2025})
    driver = pairs.assign(avg_velocity_gap_from_prev=gap, repeat_pct=repeat)
    ps = pairs.assign(whiff_diff_adj_shrunk=residual)

    result = sequencing_feature_overlap(driver, ps)

    assert result["between"] < -0.9                                     # built from overlapping pieces
    assert result["gap_whiff"] == pytest.approx(np.corrcoef(gap, residual)[0, 1])
    assert result["repeat_whiff"] == pytest.approx(np.corrcoef(repeat, residual)[0, 1])


def test_sequencing_feature_overlap_skips_pitcher_seasons_missing_a_feature_or_the_residual():
    driver = pd.DataFrame({"pitcher": [1, 2, 3, 4], "season": 2025,
                           "avg_velocity_gap_from_prev": [1.0, 2.0, 3.0, np.nan], "repeat_pct": [0.5, 0.4, 0.3, 0.2]})
    ps = pd.DataFrame({"pitcher": [1, 2, 3, 4], "season": 2025, "whiff_diff_adj_shrunk": [0.1, 0.2, np.nan, 0.4]})

    result = sequencing_feature_overlap(driver, ps)

    assert result["between"] == pytest.approx(-1.0)                      # only pitchers 1 and 2 remain
