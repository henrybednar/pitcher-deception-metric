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


def test_fmt_p_keeps_three_decimals_below_the_five_percent_line_so_a_gate_at_one_percent_reads_clearly():
    from export_site_stats import fmt_p

    assert fmt_p(0.0119) == "0.012"       # used to print "0.01", which read as passing a p < 0.01 gate
    assert fmt_p(0.007) == "0.007"
    assert fmt_p(0.16) == "0.16"
    assert fmt_p(0.0005) == "0.0005"
    assert fmt_p(0.00003) == "<0.0001"


def test_ground_ball_forecast_correlation_pairs_2025_deception_with_2026_ground_ball_by_pitcher():
    from export_site_stats import ground_ball_forecast_correlation

    ps = pd.DataFrame({
        "pitcher": [1, 2, 3, 1, 2, 3, 4],
        "season": [2025, 2025, 2025, 2026, 2026, 2026, 2026],
        "deception_plus": [90.0, 100.0, 110.0, np.nan, np.nan, np.nan, np.nan],
        "gb_index": [np.nan, np.nan, np.nan, 110.0, 100.0, 90.0, 120.0],   # pitcher 4 has no 2025 row
    })

    assert ground_ball_forecast_correlation(ps) == pytest.approx(-1.0)


def test_unscored_pitch_share_counts_pitches_with_no_pitch_type(tmp_path):
    from export_site_stats import unscored_pitch_share

    path = tmp_path / "preds.csv"
    pd.DataFrame({"pitch_type": ["FF", "SL", None, "CH"]}).to_csv(path, index=False)

    assert unscored_pitch_share(str(path)) == pytest.approx(0.25)


def test_clear_of_average_share_counts_qualified_scores_whose_interval_excludes_100():
    from export_site_stats import clear_of_average_counts

    ps = pd.DataFrame({
        "qualified": [True, True, True, True, False],
        "deception_plus": [120.0, 80.0, 101.0, 99.0, 150.0],
        "deception_plus_ci_lo": [110.0, 70.0, 90.0, 90.0, 140.0],
        "deception_plus_ci_hi": [130.0, 90.0, 112.0, 99.5, 160.0],   # row 4 stops just short of 100
    })

    assert clear_of_average_counts(ps) == (3, 4)   # the unqualified row is ignored, and the 90-112 interval spans 100
