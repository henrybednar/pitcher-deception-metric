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


def test_design_effect_is_one_when_pitches_within_a_game_are_independent():
    tbl = clustered_games(rho=0.0)

    assert rc.estimate_design_effect(tbl, rc.pitcher_season_point_estimate(tbl)) < 1.1


def test_design_effect_detects_within_game_clustering():
    tbl = clustered_games(rho=0.05)          # true design effect about 1 + 14 x 0.05 = 1.7

    assert 1.4 < rc.estimate_design_effect(tbl, rc.pitcher_season_point_estimate(tbl)) < 2.0


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
