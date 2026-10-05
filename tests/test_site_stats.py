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


def test_ground_ball_forecast_correlation_pairs_a_season_s_deception_with_the_next_season_s_ground_ball_by_pitcher():
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


def stability_frame(n=240, seed=0):
    rng = np.random.default_rng(seed)
    talent = rng.normal(100, 8, n)
    volume = np.where(np.arange(n) % 2 == 0, 2.0, 0.5)           # even pitchers throw four times as much
    rows = []
    for season in (2025, 2026):
        noise = rng.normal(0, 10, n) / np.sqrt(volume)             # more volume, less noise
        row = pd.DataFrame({"pitcher": np.arange(n), "season": season, "qualified": True,
                            "deception_plus": talent + noise})
        for label in ("whiff", "chase", "weak", "timing", "whiffmiss"):
            row[f"{label}_n"] = 100.0 * volume
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def test_year_over_year_summary_gives_an_interval_and_splits_by_volume():
    from export_site_stats import year_over_year_summary

    summary = year_over_year_summary(stability_frame(), n_boot=300)

    assert summary["n"] == 240
    assert summary["lo"] < summary["r"] < summary["hi"]
    assert summary["high_volume_r"] > summary["low_volume_r"]      # less noise, more stable


def test_year_over_year_summary_pools_every_pair_of_consecutive_seasons_and_reports_each():
    from export_site_stats import year_over_year_summary

    ps = pd.concat([stability_frame(n=100, seed=0).assign(season=lambda d: d["season"].map({2025: 2024, 2026: 2025})),
                    stability_frame(n=100, seed=0).query("season == 2026")], ignore_index=True)
    # every pitcher has 2024, 2025 and 2026: two pairs each, 200 in all

    summary = year_over_year_summary(ps, n_boot=100)

    assert summary["n"] == 200
    assert set(summary["by_pair"]) == {"2024-2025", "2025-2026"}


def test_year_over_year_summary_ignores_unqualified_and_single_season_pitchers():
    from export_site_stats import year_over_year_summary

    ps = stability_frame(n=60)
    ps.loc[(ps["pitcher"] < 10) & (ps["season"] == 2026), "qualified"] = False
    ps = ps[~((ps["pitcher"] >= 50) & (ps["season"] == 2025))]

    assert year_over_year_summary(ps, n_boot=50)["n"] == 40


def test_stuff_plus_interval_brackets_the_correlation_and_resamples_whole_pitchers():
    from export_site_stats import stuff_plus_correlation_interval

    rng = np.random.default_rng(1)
    stuff = rng.normal(100, 10, 200)
    ps = pd.concat([pd.DataFrame({"pitcher": np.arange(200), "season": s, "qualified": True, "stuff_plus": stuff,
                                  "deception_plus": 100 + 0.4 * (stuff - 100) + rng.normal(0, 9, 200)}) for s in (2025, 2026)])

    lo, hi = stuff_plus_correlation_interval(ps, n_boot=300)
    r = ps["deception_plus"].corr(ps["stuff_plus"])

    assert lo < r < hi
    assert hi - lo < 0.3


def test_handedness_gap_is_left_minus_right_with_a_welch_test():
    from export_site_stats import handedness_gap

    rng = np.random.default_rng(2)
    ps = pd.DataFrame({"pitcher": np.arange(400), "qualified": True,
                       "deception_plus": np.concatenate([rng.normal(102, 10, 100), rng.normal(100, 10, 300)])})
    hands = pd.Series(["L"] * 100 + ["R"] * 300, index=np.arange(400))

    gap = handedness_gap(ps, hands)

    assert gap["gap"] == pytest.approx(ps["deception_plus"][:100].mean() - ps["deception_plus"][100:].mean())
    assert (gap["n_left"], gap["n_right"]) == (100, 300)
    assert 0.0 < gap["p"] < 1.0


def test_membership_text_formats_each_variant_with_its_difference_and_interval():
    from export_site_stats import membership_text

    entry = {"reliability": 0.7114, "yoy_all": 0.5936, "diff": -0.0266, "diff_lo": -0.0441, "diff_hi": -0.0098}
    check = {name: entry for name in ("+gb", "+align", "+calledstrike", "-whiffmiss", "-weak", "-timing")}

    text = membership_text(check)

    assert text["MEM_GB_REL"] == "0.711" and text["MEM_GB_YOY"] == "0.594"
    assert text["MEM_CS_DIFF"] == "-0.027" and text["MEM_CS_CI"] == "-0.044 to -0.010"
    assert set(text) >= {"MEM_NO_WHIFFMISS_DIFF", "MEM_NO_WEAK_CI", "MEM_NO_TIMING_REL", "MEM_ALIGN_YOY"}
