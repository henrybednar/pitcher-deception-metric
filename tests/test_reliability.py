import numpy as np
import pandas as pd
import pytest

import reliability_and_ci as rc


def game_table(expected_means: list[float], n_per_game: int = 20) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(expected_means):
        rows.append({"pitcher": i, "season": 2025, "game_pk": 1, "n": n_per_game,
                     "actual_sum": 0.0, "expected_sum": p * n_per_game,
                     "var_sum": float(rc.binary_pitch_variance(pd.Series([p])).iloc[0]) * n_per_game})
    return pd.DataFrame(rows)


def test_pitch_variance_is_clipped_for_saturated_predictions():
    variance = rc.binary_pitch_variance(pd.Series([0.0, 1.0, 0.3]))

    # predicted rates are clipped to 0.01-0.99 (README: Fixes), so a saturated prediction still carries variance
    assert variance.tolist() == pytest.approx([0.01 * 0.99, 0.01 * 0.99, 0.3 * 0.7])


def test_season_variance_is_the_summed_pitch_variance_over_n_squared():
    agg = rc.pitcher_season_point_estimate(game_table([0.05, 0.5, 0.95]))

    expected = [p * (1 - p) / 20 for p in (0.05, 0.5, 0.95)]
    assert agg["sampling_var"].tolist() == pytest.approx(expected)


def test_pitch_level_variance_is_below_the_pooled_formula_when_predictions_vary_within_a_season():
    expected = pd.Series([0.1] * 50 + [0.5] * 50)

    pitch_level = rc.binary_pitch_variance(expected).sum() / len(expected) ** 2
    pooled = expected.mean() * (1 - expected.mean()) / len(expected)

    assert pitch_level == pytest.approx(0.0017)
    assert pitch_level < pooled


def test_point_estimate_without_a_variance_column_still_returns_the_means():
    tbl = game_table([0.3]).drop(columns="var_sum")

    agg = rc.pitcher_season_point_estimate(tbl)

    assert agg.loc[0, "expected_mean"] == pytest.approx(0.3)
    assert np.isnan(agg.loc[0, "sampling_var"])


def test_shrinkage_pulls_a_noisy_estimate_toward_zero_and_scales_to_100():
    rng = np.random.default_rng(1)
    n = 400
    agg = pd.DataFrame({"pitcher": range(n), "n": np.full(n, 200),
                        "diff": rng.normal(0, 0.08, n), "sampling_var": np.full(n, 0.002)})

    shrunk, index, league_std, true_var, center = rc.shrink_and_scale(agg)

    assert np.abs(shrunk).max() < np.abs(agg["diff"]).max()
    assert index.mean() == pytest.approx(100, abs=1.0)
    assert index.std() == pytest.approx(10, abs=0.05)
    assert true_var > 0 and league_std > 0


def test_true_variance_is_the_floor_when_residuals_are_tighter_than_the_sampling_noise():
    diff = np.zeros(50)
    sampling_var = np.full(50, 0.01)

    assert rc.estimate_true_var(diff, sampling_var) == pytest.approx(rc.TAU2_FLOOR)


def test_true_variance_recovers_the_between_pitcher_variance():
    rng = np.random.default_rng(11)
    k = 4000
    sampling_var = rng.uniform(0.0002, 0.001, k)
    diff = rng.normal(0, 0.02, k) + rng.normal(0, np.sqrt(sampling_var))

    assert rc.estimate_true_var(diff, sampling_var) == pytest.approx(0.02 ** 2, rel=0.15)


def test_winsorizing_bounds_the_damage_from_a_few_extreme_pitcher_seasons():
    rng = np.random.default_rng(12)
    k = 1500
    sampling_var = rng.uniform(0.001, 0.01, k)
    diff = rng.normal(0, 0.02, k) + rng.normal(0, np.sqrt(sampling_var))
    clean = rc.estimate_true_var(diff, sampling_var)

    diff_out = np.concatenate([diff, np.full(5, 1.0)])
    var_out = np.concatenate([sampling_var, np.full(5, 0.0003)])
    winsorized = rc.estimate_true_var(diff_out, var_out)
    unclipped = rc.estimate_true_var(diff_out, var_out, clip=np.inf)

    assert winsorized < 2.5 * clean
    assert winsorized < 0.2 * unclipped


def test_shrink_and_scale_estimates_tau2_from_collapsed_repeated_pitchers():
    rng = np.random.default_rng(41)
    k = 200
    sampling_var = np.full(k, 0.002)
    talent = rng.normal(0, 0.03, k)
    diff_2025 = talent + rng.normal(0, np.sqrt(sampling_var))
    diff_2026 = talent + rng.normal(0, np.sqrt(sampling_var))
    pitcher = np.arange(k)
    agg = pd.DataFrame({
        "pitcher": np.concatenate([pitcher, pitcher]),
        "n": np.full(2 * k, 200),
        "diff": np.concatenate([diff_2025, diff_2026]),
        "sampling_var": np.full(2 * k, 0.002),
    })

    _, _, _, true_var, _ = rc.shrink_and_scale(agg)

    collapsed_diff, collapsed_sv = rc.collapse_repeated_pitchers(
        agg["pitcher"].values, agg["n"].values, agg["diff"].values, agg["sampling_var"].values)
    expected_true_var = rc.estimate_true_var(collapsed_diff, collapsed_sv)
    naive_true_var = rc.estimate_true_var(agg["diff"].values, agg["sampling_var"].values)

    assert true_var == pytest.approx(expected_true_var)
    # A naive tau2 over the uncollapsed 2k rows is a different estimate: duplicating each pitcher's
    # talent draw across two rows is not the same amount of independent information as 2k distinct
    # pitchers, so shrink_and_scale must not compute tau2 from the uncollapsed rows.
    assert true_var != pytest.approx(naive_true_var, rel=0.01)


def test_collapse_repeated_pitchers_is_a_noop_when_no_pitcher_repeats():
    diff = np.array([0.1, -0.2, 0.3])
    sampling_var = np.array([0.01, 0.02, 0.03])

    collapsed_diff, collapsed_sv = rc.collapse_repeated_pitchers(
        np.array([1, 2, 3]), np.array([100, 100, 100]), diff, sampling_var)

    assert collapsed_diff.tolist() == pytest.approx(diff.tolist())
    assert collapsed_sv.tolist() == pytest.approx(sampling_var.tolist())


def clustered_games(rho: float, n_seasons: int = 150, n_games: int = 20, mean_m: int = 15, p: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    rows = []
    for season_id in range(n_seasons):
        m = np.maximum(rng.poisson(mean_m, n_games), 1)
        p_game = rng.beta(p * (1 - rho) / rho, (1 - p) * (1 - rho) / rho, n_games) if rho else np.full(n_games, p)
        wins = rng.binomial(m, p_game)
        for g in range(n_games):
            rows.append(dict(pitcher=season_id, season=2025, game_pk=g, n=m[g], actual_sum=wins[g], expected_sum=p * m[g],
                             var_sum=p * (1 - p) * m[g]))
    return pd.DataFrame(rows)


def test_design_effect_is_near_one_when_pitches_within_a_game_are_independent():
    tbl = clustered_games(rho=0.0)

    curve = rc.estimate_design_effect_curve(tbl, rc.pitcher_season_point_estimate(tbl))

    assert all(1.0 <= effect < 1.15 for _, effect in curve)                   # never below 1, even when noise makes the ratio dip under it


def test_design_effect_detects_within_game_clustering():
    tbl = clustered_games(rho=0.05)          # true design effect about 1 + 14 x 0.05 = 1.7

    curve = rc.estimate_design_effect_curve(tbl, rc.pitcher_season_point_estimate(tbl))

    assert all(1.3 < effect < 2.1 for _, effect in curve)


def test_design_effect_corrects_the_bootstrap_for_few_games():
    tbl = clustered_games(rho=0.05, n_seasons=600, n_games=6)          # a bootstrap over 6 games understates the variance by 5/6; true factor about 1.7

    curve = rc.estimate_design_effect_curve(tbl, rc.pitcher_season_point_estimate(tbl), n_bins=1)

    assert 1.55 < curve[0][1] < 1.9                                      # without the g / (g - 1) correction it would come out near 1.42


def test_a_series_design_effect_is_matched_to_scored_rows_by_index_not_position():
    agg = pd.DataFrame({"pitcher": np.arange(40), "season": 2025, "n": [10] * 5 + [200] * 35, "n_games": 10,
                        "diff": np.linspace(-0.05, 0.05, 40), "sampling_var": 0.0004})
    effect = pd.Series([1.0] * 5 + [2.0] * 35, index=agg.index)          # the first five rows are too small to score

    by_series = rc.shrink_and_scale(agg, effect)[0]
    by_scalar = rc.shrink_and_scale(agg, 2.0)[0]

    assert by_series.iloc[5:].tolist() == pytest.approx(by_scalar.iloc[5:].tolist())
    assert by_series.iloc[:5].isna().all()


def test_design_effect_curve_rises_with_the_pitches_per_game():
    few, many = clustered_games(rho=0.05, mean_m=4), clustered_games(rho=0.05, mean_m=40)
    many["pitcher"] += 1000
    tbl = pd.concat([few, many], ignore_index=True)

    curve = rc.estimate_design_effect_curve(tbl, rc.pitcher_season_point_estimate(tbl), n_bins=2)

    (m_low, effect_low), (m_high, effect_high) = sorted(curve)
    assert m_low < 6 and m_high > 30
    assert effect_low < 1.4 < effect_high


def test_design_effect_for_reads_the_curve_at_each_rows_cluster_size_and_holds_flat_beyond_it():
    agg = pd.DataFrame({"n": [20, 100, 400, 1000], "n_games": [10, 10, 10, 10]})            # 2, 10, 40 and 100 pitches per game
    curve = [(5.0, 1.0), (15.0, 1.2), (45.0, 1.8)]

    effect = rc.design_effect_for(agg, curve)

    assert effect.tolist() == pytest.approx([1.0, 1.1, 1.7, 1.8])                       # flat below 5, interpolated between bins, flat above 45


def test_design_effect_curve_of_one_point_is_a_constant_factor():
    agg = pd.DataFrame({"n": [20, 400], "n_games": [10, 10]})

    assert rc.design_effect_for(agg, [(10.0, 1.3)]).tolist() == [1.3, 1.3]


def test_a_larger_cluster_size_inflates_the_sampling_variance_and_so_shrinks_a_score_further():
    n = 200
    agg = pd.DataFrame({"pitcher": np.arange(60), "season": 2025, "n": n, "n_games": 10, "diff": np.linspace(-0.05, 0.05, 60),
                        "sampling_var": 0.0004})
    effect = rc.design_effect_for(agg, [(20.0, 1.38)])                 # 20 pitches per game: factor 1.38
    flat = rc.shrink_and_scale(agg, 1.0)[0]
    inflated = rc.shrink_and_scale(agg, effect)[0]

    assert effect.iloc[0] == pytest.approx(1.38)
    assert inflated.abs().mean() < flat.abs().mean()
    assert rc.shrink_and_scale(agg, 1.38)[0].tolist() == pytest.approx(inflated.tolist())    # a Series of equal factors matches the scalar


def test_shrinkage_targets_the_grand_mean_so_a_calibration_bias_moves_the_scale_not_the_pitchers():
    rng = np.random.default_rng(2)
    n = 500
    biased = pd.DataFrame({"pitcher": range(n), "n": np.full(n, 200),
                           "diff": rng.normal(0.05, 0.08, n), "sampling_var": np.full(n, 0.002)})

    _, index, _, _, center = rc.shrink_and_scale(biased)

    assert center == pytest.approx(0.05, abs=0.01)
    assert index.mean() == pytest.approx(100, abs=1.0)


def synthetic_pitches(saturated: bool) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for pitcher in range(1, 121):
        for i in range(250):
            game = 1000 + pitcher * 100 + i % 25
            rows.append(dict(pitcher=pitcher, season=2025, game_pk=game, half=game % 2, pitch_type="FF", is_swing=True,
                             is_in_zone=False, is_bip=False, is_whiff=int(rng.random() < 0.25), whiff_expected_full=0.25))
    if saturated:
        for i in range(30):
            game = 900000 + i % 3
            rows.append(dict(pitcher=9001, season=2025, game_pk=game, half=game % 2, pitch_type="FF", is_swing=True,
                             is_in_zone=False, is_bip=False, is_whiff=1, whiff_expected_full=0.25))
    return pd.DataFrame(rows)


def test_a_saturated_thirty_pitch_sample_still_gets_an_interval_with_width():
    df = synthetic_pitches(saturated=True)

    result, _, _ = rc.process_outcome(df, "whiff", rc.BINARY_OUTCOMES["whiff"], is_binary=True)

    row = result.set_index("pitcher").loc[9001]
    assert row["whiff_n"] == 30
    assert row["whiff_ci_hi"] - row["whiff_ci_lo"] > 5          # index points; a bootstrap of 30/30 alone gives 0
    assert row["whiff_ci_lo"] <= row["whiff_index"] <= row["whiff_ci_hi"]


def test_pitches_without_an_expectation_do_not_count_toward_n():
    df = synthetic_pitches(saturated=False)
    df.loc[df.pitcher == 1, "whiff_expected_full"] = np.nan

    result, _, _ = rc.process_outcome(df, "whiff", rc.BINARY_OUTCOMES["whiff"], is_binary=True)

    assert 1 not in result.loc[result["whiff_index"].notna(), "pitcher"].tolist()


def test_pitcher_seasons_under_three_games_get_an_analytic_interval():
    df = synthetic_pitches(saturated=False)
    one_game = df[df.pitcher == 2].head(40).copy()
    one_game["pitcher"], one_game["game_pk"], one_game["half"] = 9002, 777777, 1
    df = pd.concat([df, one_game], ignore_index=True)

    result, _, _ = rc.process_outcome(df, "whiff", rc.BINARY_OUTCOMES["whiff"], is_binary=True)

    row = result.set_index("pitcher").loc[9002]
    assert np.isfinite(row["whiff_ci_lo"]) and np.isfinite(row["whiff_ci_hi"])


def test_a_component_with_no_talent_is_flat_at_100_instead_of_amplified_to_a_spread_of_10():
    rng = np.random.default_rng(31)
    k = 1500
    sampling_var = rng.uniform(0.001, 0.01, k)
    agg = pd.DataFrame({"pitcher": range(k), "n": np.full(k, 200),
                        "diff": rng.normal(0, np.sqrt(sampling_var)), "sampling_var": sampling_var})

    shrunk, index, league_std, true_var, _ = rc.shrink_and_scale(agg)

    assert not rc.has_between_pitcher_spread(agg["diff"].values, sampling_var)
    assert (index == 100).all() and (shrunk == 0).all()
    assert true_var == rc.TAU2_FLOOR


def test_between_pitcher_spread_is_detected_when_it_exists():
    rng = np.random.default_rng(32)
    k = 1500
    sampling_var = rng.uniform(0.001, 0.01, k)
    diff = rng.normal(0, 0.03, k) + rng.normal(0, np.sqrt(sampling_var))

    assert rc.has_between_pitcher_spread(diff, sampling_var)


def test_robust_std_ignores_a_few_extreme_values_but_matches_the_std_of_clean_data():
    rng = np.random.default_rng(33)
    clean = rng.normal(0, 1, 5000)
    dirty = np.concatenate([clean, np.full(5, 40.0)])

    assert rc.robust_std(clean) == pytest.approx(clean.std(), rel=0.01)
    assert rc.robust_std(dirty) == pytest.approx(clean.std(), rel=0.05)
    assert dirty.std() > 1.5 * clean.std()


def composite_frame(index_a, index_b, n_a=100, n_b=100):
    return pd.DataFrame({"a_index": index_a, "b_index": index_b, "a_n": n_a, "b_n": n_b})


def test_composite_is_the_reliability_weighted_mean_of_the_component_z_scores():
    frame = composite_frame([120.0], [80.0])

    raw, n_comp, eligible = rc.compute_composite(frame, ["a_index", "b_index"], ["a_n", "b_n"], [0.6, 0.3])

    assert raw[0] == pytest.approx((0.6 * 2 + 0.3 * -2) / 0.9)
    assert n_comp[0] == 2 and eligible[0]


def test_a_component_below_the_minimum_reliability_gets_no_weight():
    frame = composite_frame([120.0], [80.0])

    raw, _, _ = rc.compute_composite(frame, ["a_index", "b_index"], ["a_n", "b_n"], [0.6, 0.1])

    assert raw[0] == pytest.approx(2.0)


def test_composite_needs_at_least_two_components_and_weights_by_relative_sample_size():
    frame = composite_frame([120.0, 110.0, 130.0], [np.nan, 90.0, 90.0], n_a=[100, 100, 300], n_b=[100, 100, 100])

    raw, n_comp, eligible = rc.compute_composite(frame, ["a_index", "b_index"], ["a_n", "b_n"], [0.5, 0.5])

    assert not eligible[0] and np.isnan(raw[0])
    assert raw[1] == pytest.approx(0.0)
    assert raw[2] > raw[1]                    # the larger a_n pulls the average toward a's positive z


def test_merge_fangraphs_columns_adds_them_when_present():
    ps = pd.DataFrame({"pitcher": [1, 2], "season": [2025, 2025]})
    covariates = pd.DataFrame({"pitcher": [1, 2], "season": [2025, 2025],
                               "stuff_plus": [110.0, 95.0], "location_plus": [105.0, 90.0], "pitching_plus": [108.0, 92.0]})

    result = rc.merge_fangraphs_columns(ps, covariates)

    assert result["stuff_plus"].tolist() == [110.0, 95.0]
    assert set(rc.FANGRAPHS_COLS) <= set(result.columns)


def test_merge_fangraphs_columns_is_all_nan_when_no_fangraphs_export_was_dropped_in():
    ps = pd.DataFrame({"pitcher": [1, 2], "season": [2025, 2025]})
    covariates = pd.DataFrame({"pitcher": [1, 2], "season": [2025, 2025]})   # no FanGraphs columns at all

    result = rc.merge_fangraphs_columns(ps, covariates)

    assert result[rc.FANGRAPHS_COLS].isna().all().all()
    assert len(result) == 2


def test_posterior_sd_is_the_normal_normal_posterior_on_the_z_scale():
    sd = rc.posterior_sd_z(np.array([0.0004, 0.0100]), design_effect=1.0, true_var=0.0004, league_std=0.01)

    # equal signal and noise halves the variance, so the sd is sqrt(0.0002) / league_std
    assert sd[0] == pytest.approx(np.sqrt(0.0002) / 0.01)
    # a noisy sample falls back toward the between-pitcher spread
    assert sd[1] == pytest.approx(np.sqrt(0.0004 * 0.01 / 0.0104) / 0.01)


def test_posterior_sd_grows_with_the_design_effect():
    base = rc.posterior_sd_z(np.array([0.001]), design_effect=1.0, true_var=0.0004, league_std=0.01)
    inflated = rc.posterior_sd_z(np.array([0.001]), design_effect=2.0, true_var=0.0004, league_std=0.01)

    assert inflated[0] > base[0]


def weights_frame():
    return pd.DataFrame({"a_index": [110.0, np.nan], "b_index": [90.0, 105.0], "c_index": [100.0, 120.0],
                         "a_n": [200, 200], "b_n": [100, 100], "c_n": [50, 50]})


def test_composite_weights_zero_out_absent_and_unreliable_components():
    frame = weights_frame()

    w = rc.composite_weights(frame, ["a_index", "b_index", "c_index"], ["a_n", "b_n", "c_n"], [0.8, 0.5, 0.1])

    assert w[1, 0] == 0.0                       # component a is missing for the second row
    assert (w[:, 2] == 0.0).all()               # component c is below the minimum reliability
    assert w[0, 0] == pytest.approx(0.8 * 200 / 200)   # reliability times n over the median n


def test_composite_weights_reproduce_the_composite_z():
    frame = weights_frame()
    cols, ns, rel = ["a_index", "b_index", "c_index"], ["a_n", "b_n", "c_n"], [0.8, 0.5, 0.6]

    w = rc.composite_weights(frame, cols, ns, rel)
    raw_z, _, eligible = rc.compute_composite(frame, cols, ns, rel)

    z = (frame[cols].fillna(100).values - 100) / 10
    assert raw_z[0] == pytest.approx((w[0] * z[0]).sum() / w[0].sum())
    assert eligible.tolist() == [True, True]


def test_composite_posterior_sd_of_equal_independent_components_shrinks_with_the_square_root():
    w = np.array([[1.0, 1.0, 1.0, 1.0]])
    sd = np.full((1, 4), 0.3)

    assert rc.composite_posterior_sd(w, sd)[0] == pytest.approx(0.3 / 2)


def test_composite_posterior_sd_ignores_components_with_no_weight():
    w = np.array([[1.0, 0.0]])
    sd = np.array([[0.3, np.nan]])

    assert rc.composite_posterior_sd(w, sd)[0] == pytest.approx(0.3)


def test_composite_interval_is_symmetric_on_the_deception_scale():
    lo, hi = rc.composite_interval(np.array([110.0]), np.array([0.2]), pop_std=0.5)

    half = 1.96 * 10 * 0.2 / 0.5
    assert (lo[0], hi[0]) == pytest.approx((110.0 - half, 110.0 + half))


def test_the_grand_mean_weights_each_pitcher_by_one_over_tau2_plus_sampling_variance():
    rng = np.random.default_rng(7)
    n = 300
    sv = rng.uniform(0.0002, 0.01, n)
    talent = rng.normal(0.01, 0.03, n)
    agg = pd.DataFrame({"pitcher": range(n), "n": np.full(n, 100), "diff": talent + rng.normal(0, np.sqrt(sv)), "sampling_var": sv})

    _, _, _, true_var, center = rc.shrink_and_scale(agg)

    weights = 1.0 / (true_var + sv)
    assert true_var > 1e-4
    assert center == pytest.approx(np.sum(weights * agg["diff"]) / np.sum(weights))
    sampling_only = np.sum(agg["diff"] / sv) / np.sum(1.0 / sv)
    assert abs(center - sampling_only) > 1e-5          # the two weightings give different answers on this data


def test_bootstrap_interval_uses_each_pitcher_seasons_own_design_effect():
    games = pd.DataFrame({"pitcher": [1] * 12 + [2] * 12, "season": 2025, "game_pk": list(range(12)) * 2, "n": 20,
                          "actual_sum": [5, 7, 4, 6, 5, 8, 4, 5, 6, 5, 7, 4] * 2, "expected_sum": 5.0, "var_sum": 3.75})

    ci = rc.bootstrap_ci(games, league_std=0.02, true_var=0.0004, design_effect={(1, 2025): 1.0, (2, 2025): 4.0}, center=0.0).set_index("pitcher")

    width = ci["ci_hi"] - ci["ci_lo"]
    assert width[2] < 0.6 * width[1]                 # same results, but pitcher 2 is shrunk much harder because its pitches are more clustered


def test_design_effect_never_falls_below_one_and_needs_enough_games():
    steady = clustered_games(rho=0.0, n_seasons=40, n_games=10).assign(actual_sum=lambda d: d["expected_sum"])      # no variation between games at all
    short = clustered_games(rho=0.05, n_seasons=40, n_games=2)

    curve_steady = rc.estimate_design_effect_curve(steady, rc.pitcher_season_point_estimate(steady))
    curve_short = rc.estimate_design_effect_curve(short, rc.pitcher_season_point_estimate(short))

    assert all(effect == 1.0 for _, effect in curve_steady)
    assert curve_short == [(1.0, 1.0)]                                   # fewer than five games per pitcher-season: nothing to estimate from


def test_posterior_sd_grows_with_a_per_row_design_effect():
    sd = rc.posterior_sd_z(np.array([0.001, 0.001]), pd.Series([1.0, 4.0]), true_var=0.0004, league_std=0.01)

    assert sd[1] > sd[0]


def test_process_outcome_applies_the_cluster_size_design_effect_to_its_report():
    rng = np.random.default_rng(8)
    rows = []
    for pitcher in range(150):
        starter = pitcher % 2 == 0
        for g in range(30 if starter else 50):
            shock = rng.normal(0, 0.05)
            for _ in range(30 if starter else 6):
                p = float(np.clip(0.25 + shock, 0.01, 0.99))
                game = pitcher * 100 + g
                rows.append(dict(pitcher=pitcher, season=2025, game_pk=game, half=game % 2, pitch_type="FF", is_swing=True, is_in_zone=False,
                                 is_bip=False, is_whiff=int(rng.random() < p), whiff_expected_full=0.25))
    df = pd.DataFrame(rows)

    result, rel, _ = rc.process_outcome(df, "whiff", rc.BINARY_OUTCOMES["whiff"], is_binary=True)

    assert rel["design_effect"] > 1.2 and len(rel["design_effect_curve"]) == 3          # 150 pitcher-seasons give three bins of 50
    effects = [effect for _, effect in sorted(rel["design_effect_curve"])]
    assert effects[-1] > effects[0] + 0.2                                  # starters (30 pitches per game) carry far more clustering than relievers (6)
    assert result["whiff_ci_hi"].gt(result["whiff_ci_lo"]).all()


def test_design_effect_curve_uses_fewer_bins_when_there_are_few_pitcher_seasons():
    tbl = clustered_games(rho=0.05, n_seasons=120, n_games=20)                  # 120 pitcher-seasons: room for 2 bins of 50, not 4

    curve = rc.estimate_design_effect_curve(tbl, rc.pitcher_season_point_estimate(tbl))

    assert len(curve) == 2
    assert len(rc.estimate_design_effect_curve(tbl.iloc[:600], rc.pitcher_season_point_estimate(tbl.iloc[:600]))) == 1       # 30 pitcher-seasons: one pooled factor


def test_the_reported_design_effect_averages_over_scored_pitcher_seasons_only():
    rng = np.random.default_rng(3)
    rows = []

    def add(pitcher, games, per_game, shock_sd):
        for g in range(games):
            shock = rng.normal(0, shock_sd)
            game = pitcher * 100 + g
            for _ in range(per_game):
                rows.append(dict(pitcher=pitcher, season=2025, game_pk=game, half=game % 2, pitch_type="FF", is_swing=True, is_in_zone=False, is_bip=False,
                                 is_whiff=int(rng.random() < np.clip(0.25 + shock, 0.01, 0.99)), whiff_expected_full=0.25))

    for pitcher in range(120):                                       # scored: 10 or 30 pitches per game, clustered within games
        add(pitcher, 25, 10 if pitcher % 2 else 30, 0.05)
    for pitcher in range(1000, 1060):                                # not scored: 6 games of 2 pitches, 12 pitches in all
        add(pitcher, 6, 2, 0.05)
    df = pd.DataFrame(rows)
    games = rc.game_level_table(df.assign(_v=rc.binary_pitch_variance(df["whiff_expected_full"])), "whiff", "is_whiff", "full", "_v")
    agg = rc.pitcher_season_point_estimate(games)

    _, rel, _ = rc.process_outcome(df, "whiff", rc.BINARY_OUTCOMES["whiff"], is_binary=True)

    every_row = rc.design_effect_for(agg, rel["design_effect_curve"]).mean()
    assert rel["design_effect"] > every_row + 0.02                    # the short, unscored pitcher-seasons would pull a mean over every row down
