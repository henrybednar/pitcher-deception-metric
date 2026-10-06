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
    driver = pairs.assign(velocity_gap_per_switch=gap, repeat_pct=repeat)
    ps = pairs.assign(whiff_diff_adj_shrunk=residual)

    result = sequencing_feature_overlap(driver, ps)

    assert result["between"] < -0.9                                     # built from overlapping pieces
    assert result["gap_whiff"] == pytest.approx(np.corrcoef(gap, residual)[0, 1])
    assert result["repeat_whiff"] == pytest.approx(np.corrcoef(repeat, residual)[0, 1])


def test_sequencing_feature_overlap_skips_pitcher_seasons_missing_a_feature_or_the_residual():
    driver = pd.DataFrame({"pitcher": [1, 2, 3, 4], "season": 2025,
                           "velocity_gap_per_switch": [1.0, 2.0, 3.0, np.nan], "repeat_pct": [0.5, 0.4, 0.3, 0.2]})
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


def test_handedness_gap_is_left_minus_right_with_a_pitcher_clustered_test():
    from export_site_stats import handedness_gap

    rng = np.random.default_rng(2)
    ps = pd.DataFrame({"pitcher": np.arange(400), "qualified": True,
                       "deception_plus": np.concatenate([rng.normal(102, 10, 100), rng.normal(100, 10, 300)])})
    hands = pd.Series(["L"] * 100 + ["R"] * 300, index=np.arange(400))

    gap = handedness_gap(ps, hands)

    assert gap["gap"] == pytest.approx(ps["deception_plus"][:100].mean() - ps["deception_plus"][100:].mean())
    assert (gap["n_left"], gap["n_right"]) == (100, 300)
    assert 0.0 < gap["p"] < 1.0


def test_handedness_p_value_is_wider_when_a_pitchers_seasons_repeat_the_same_score():
    from export_site_stats import handedness_gap

    rng = np.random.default_rng(4)
    talent = np.concatenate([rng.normal(101, 6, 60), rng.normal(100, 6, 180)])
    hands = pd.Series(["L"] * 60 + ["R"] * 180, index=np.arange(240))
    three = pd.concat([pd.DataFrame({"pitcher": np.arange(240), "season": s, "qualified": True, "deception_plus": talent + rng.normal(0, 0.5, 240)})
                       for s in (2024, 2025, 2026)], ignore_index=True)
    one = three[three["season"] == 2024]

    assert handedness_gap(three, hands)["p"] > handedness_gap(one, hands)["p"] * 0.9      # tripling the rows does not triple the evidence
    assert handedness_gap(three, hands)["n_left"] == 180


def test_membership_text_formats_each_variant_with_its_difference_and_interval():
    from export_site_stats import membership_text

    entry = {"reliability": 0.7114, "yoy_all": 0.5936, "diff": -0.0266, "diff_lo": -0.0441, "diff_hi": -0.0098}
    check = {name: entry for name in ("+gb", "+align", "+calledstrike", "-whiffmiss", "-weak", "-timing")}

    text = membership_text(check)

    assert text["MEM_GB_REL"] == "0.711" and text["MEM_GB_YOY"] == "0.594"
    assert text["MEM_CS_DIFF"] == "-0.027" and text["MEM_CS_CI"] == "-0.044 to -0.010"
    assert set(text) >= {"MEM_NO_WHIFFMISS_DIFF", "MEM_NO_WEAK_CI", "MEM_NO_TIMING_REL", "MEM_ALIGN_YOY"}


def test_projection_text_reports_the_errors_and_whether_the_second_season_gain_includes_zero():
    from export_site_stats import projection_text

    base = {"backtest": {"pairs": 1089, "triples": 402, "rmse_projection": 7.28, "rmse_raw_score": 8.29, "rmse_league_average": 9.28,
                         "coverage_80": 0.8, "triples_rmse_one_season": 7.101, "triples_rmse_two_seasons": 7.01,
                         "two_season_gain": 0.095, "two_season_gain_lo": -0.043, "two_season_gain_hi": 0.229},
            "coefficients": {"one_season": {"intercept": 40.0, "this": 0.594}, "two_seasons": {"intercept": 31.8, "this": 0.489, "previous": 0.182}},
            "latest_season": 2026}

    text = projection_text(base)

    assert text["PROJ_NEXT_SEASON"] == "2027" and text["PROJ_N"] == "1,089" and text["PROJ_RMSE"] == "7.3"
    assert "includes zero" in text["PROJ_TWO_NOTE"]
    assert text["PROJ_COEF_TWO"].startswith("31.8 + 0.49")
    base["backtest"]["two_season_gain_lo"] = 0.02
    assert "excludes zero" in projection_text(base)["PROJ_TWO_NOTE"]


def test_check_page_text_refuses_nan_and_inf_figures_but_not_words_that_contain_them():
    from export_site_stats import check_page_text

    check_page_text({"TOP1_NAME": "Infante, Gregory", "X": "Hannan 0.5", "R": "0.610"}, [{"key": "whiff", "label": "Whiff", "p": "<0.0001"}])
    with pytest.raises(ValueError, match="STUFF_R"):
        check_page_text({"STUFF_R": "nan"}, [])
    with pytest.raises(ValueError, match="whiff.yoy"):
        check_page_text({}, [{"key": "whiff", "yoy": "inf"}])


def test_sequencing_survivors_lists_the_outcomes_where_each_feature_clears_the_correction():
    from driver_features import FEATURE_LABELS
    from export_site_stats import sequencing_survivors

    gap, repeat = FEATURE_LABELS["velocity_gap_per_switch"], FEATURE_LABELS["repeat_pct"]
    analysis = {
        "whiff": {"features": [{"feature": gap, "q": 0.001}, {"feature": repeat, "q": 0.4}]},
        "weak": {"features": [{"feature": gap, "q": 0.02}, {"feature": repeat, "q": 0.5}]},
        "timing": {"features": [{"feature": gap, "q": 0.2}, {"feature": repeat, "q": 0.9}]},
        "align": {"features": [{"feature": gap, "q": 0.6}, {"feature": repeat, "q": 0.03}]},
    }

    out = sequencing_survivors(analysis)

    assert out["SEQ_GAP_SURVIVES"] == "whiff and weak contact"
    assert out["SEQ_REPEAT_SURVIVES"] == "horizontal alignment"
    assert sequencing_survivors({"whiff": {"features": [{"feature": gap, "q": 0.5}]}})["SEQ_GAP_SURVIVES"] == "none of the outcomes"


def test_speed_spread_correlation_compares_the_gap_per_switch_with_the_pitch_types_speed_spread():
    from export_site_stats import speed_spread_correlation

    # three pitchers with two pitch types each; the spread is the pitch-weighted SD of the two average speeds
    pitch_types = pd.DataFrame({
        "pitcher": [1, 1, 2, 2, 3, 3, 4, 4], "season": 2025, "pitches": [50, 50, 50, 50, 50, 50, 50, 5],
        "release_speed_mean": [95.0, 85.0, 95.0, 90.0, 95.0, 93.0, 95.0, 70.0]})
    driver = pd.DataFrame({"pitcher": [1, 2, 3, 4], "season": 2025, "velocity_gap_per_switch": [10.0, 5.0, 2.0, 1.0]})

    assert speed_spread_correlation(driver, pitch_types) == pytest.approx(1.0)   # pitcher 4's second type is under 20 pitches, so no spread


def test_location_text_reports_the_correlations_and_the_member_regression_coefficients():
    from export_site_stats import location_text

    rng = np.random.default_rng(1)
    location = rng.normal(100, 10, 300)
    ps = pd.DataFrame({"qualified": True, "location_plus": location, "deception_plus": 100 - location + rng.normal(0, 1, 300),
                       "stuff_plus": rng.normal(100, 10, 300), "whiffmiss_index": 100 - location, "whiff_index": 100 - location, "chase_index": 100 - location})
    ps.loc[:9, "qualified"] = False                                       # unqualified rows stay out of the correlations
    ps.loc[10, "location_plus"] = np.nan                                  # so do rows with no Location+
    validation = {"coefficients": {"k_pct": {"location_plus": {"p": 0.83}},
                                   "xwoba": {"location_plus": {"coef": -0.0042, "lo": -0.0061, "hi": -0.0023, "p": 0.0000004}}}}

    out = location_text(ps, validation)

    assert float(out["LOC_DECEPTION_R"]) < -0.9
    assert float(out["LOC_WHIFFMISS_R"]) == pytest.approx(-1.0)
    assert out["LOC_XWOBA"].startswith("-4.2 points of xwOBA allowed (lower is better; 95% interval -6.1 to -2.3, p <0.0001")
    assert out["LOC_K_P"] == "0.83"


def test_pitch_type_text_gives_the_reliability_range_and_the_best_and_worst_types():
    from export_site_stats import pitch_type_text

    summary = {"types": {
        "CH": {"whiff": {"reliability": 0.76, "median_n": 93.0}, "chase": {"reliability": 0.64, "median_n": 101.0}},
        "SI": {"whiff": {"reliability": 0.39, "median_n": 101.0}, "chase": {"reliability": 0.40, "median_n": 94.0}},
        "KC": {"whiff": {"reliability": None, "median_n": 110.0}, "chase": {"reliability": 0.63, "median_n": 145.0}},
    }}

    out = pitch_type_text(summary)

    assert out["PT_WHIFF_REL_RANGE"] == "0.39 to 0.76"                         # a type with no estimate is left out of the range
    assert out["PT_WHIFF_BEST"] == "changeups (0.76)" and out["PT_WHIFF_WORST"] == "sinkers (0.39)"
    assert out["PT_CHASE_REL_RANGE"] == "0.40 to 0.64"
    assert out["PT_MEDIAN_N"] == "101"
