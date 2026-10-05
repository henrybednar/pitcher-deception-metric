import numpy as np
import pandas as pd
import pytest

import outcome_validation as ov


def raw_chunk():
    return pd.DataFrame({
        "pitcher": [1, 1, 1, 1, 1, 2],
        "season": [2025] * 6,
        "game_type": ["R", "R", "R", "R", "S", "R"],
        "events": [None, "strikeout", "walk", "single", "strikeout", "field_out"],
        "description": ["swinging_strike", "swinging_strike", "ball", "hit_into_play", "swinging_strike", "hit_into_play"],
        "woba_value": [np.nan, 0.0, 0.69, 0.88, 0.0, 0.0],
        "woba_denom": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        "estimated_woba_using_speedangle": [np.nan, np.nan, np.nan, 0.40, np.nan, 0.10],
        "delta_pitcher_run_exp": [0.01, 0.2, -0.3, -0.5, 0.2, 0.05],
    })


def test_chunk_totals_counts_plate_appearances_and_uses_the_expected_woba_only_when_it_exists():
    totals = ov.chunk_totals(raw_chunk()).loc[(1, 2025)]

    assert totals["pa"] == 3 and totals["k"] == 1 and totals["bb"] == 1          # the spring-training strikeout does not count
    assert totals["pitches"] == 4 and totals["swing"] == 3 and totals["whiff"] == 2
    assert totals["woba_num"] == pytest.approx(0.0 + 0.69 + 0.88) and totals["woba_den"] == 3.0
    assert totals["xwoba_num"] == pytest.approx(0.0 + 0.69 + 0.40)               # the single gets its expected value, the walk keeps its own
    assert totals["rv"] == pytest.approx(0.01 + 0.2 - 0.3 - 0.5)


def test_outcome_rates_combine_chunks_of_the_same_pitcher_season_before_dividing():
    first, second = ov.chunk_totals(raw_chunk().iloc[:3]), ov.chunk_totals(raw_chunk().iloc[3:])

    rates = ov.outcome_rates(pd.concat([first, second])).set_index("pitcher").loc[1]

    assert rates["pa"] == 3 and rates["k_pct"] == pytest.approx(1 / 3) and rates["bb_pct"] == pytest.approx(1 / 3)
    assert rates["xwoba"] == pytest.approx((0.69 + 0.40) / 3) and rates["whiff_rate"] == pytest.approx(2 / 3)


def synthetic(n=400, seed=0, deception_matters=True):
    rng = np.random.default_rng(seed)
    talent = rng.normal(0, 1, n)
    stuff = 100 + 8 * rng.normal(0, 1, n)
    rows, outs = [], []
    for season in (2024, 2025):
        rows.append(pd.DataFrame({"pitcher": np.arange(n), "season": season, "qualified": True, "stuff_plus": stuff + rng.normal(0, 1, n),
                                  "location_plus": 100 + rng.normal(0, 8, n), "deception_plus": 100 + 8 * (talent + rng.normal(0, 0.5, n)),
                                  **{f"{m}_index": 100 + 8 * (talent + rng.normal(0, 1, n)) for m in ov.MEMBERS}}))
        k = 0.22 + 0.02 * (stuff - 100) / 8 + (0.02 * talent if deception_matters else 0.0) + rng.normal(0, 0.01, n)
        outs.append(pd.DataFrame({"pitcher": np.arange(n), "season": season, "pa": 400, "k_pct": k, "bb_pct": 0.08 + rng.normal(0, 0.01, n),
                                  "woba": 0.31 + rng.normal(0, 0.02, n), "xwoba": 0.31 - 0.5 * (k - 0.22) + rng.normal(0, 0.01, n),
                                  "whiff_rate": 0.25 + rng.normal(0, 0.02, n), "rv100": rng.normal(0, 1, n)}))
    return pd.concat(rows, ignore_index=True), pd.concat(outs, ignore_index=True)


def test_outcome_pairs_set_a_seasons_scores_against_the_next_seasons_results_and_drop_small_samples():
    ps, outcomes = synthetic(n=50)
    outcomes.loc[(outcomes["pitcher"] < 5) & (outcomes["season"] == 2025), "pa"] = 100

    pairs = ov.outcome_pairs(ps, outcomes)

    assert len(pairs) == 45 and (pairs["season"] == 2024).all()                      # only 2024 -> 2025, and 5 pitchers fell under the batters-faced floor
    assert pairs["next_k_pct"].to_numpy() == pytest.approx(outcomes[outcomes["season"] == 2025].set_index("pitcher").loc[pairs["pitcher"], "k_pct"].to_numpy())


def test_out_of_fold_predictions_never_use_a_pitchers_own_outcome():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(60), 2)
    X = rng.normal(size=(120, 2))
    y = X[:, 0] + rng.normal(size=120)
    base = ov.oof_predictions(X, y, groups, seeds=(0,))

    altered = y.copy()
    altered[groups == 3] += 1000.0
    after = ov.oof_predictions(X, altered, groups, seeds=(0,))

    assert after[groups == 3] == pytest.approx(base[groups == 3], abs=1e-9)
    assert not np.allclose(after[groups != 3], base[groups != 3])


def test_deception_adds_cross_validated_r2_when_it_carries_signal_and_does_not_when_it_is_noise():
    ps, outcomes = synthetic(n=500, seed=2)
    pairs = ov.outcome_pairs(ps, outcomes)
    signal = ov.evaluate_outcome(pairs, "k_pct")

    ps0, outcomes0 = synthetic(n=500, seed=2, deception_matters=False)
    pairs0 = ov.outcome_pairs(ps0, outcomes0)
    noise = ov.evaluate_outcome(pairs0, "k_pct")

    assert signal["deception_over_stuff_location"]["lo"] > 0
    assert noise["deception_over_stuff_location"]["gain"] < 0.01 and noise["deception_over_stuff_location"]["lo"] < 0.01
    assert signal["r2"]["stuff_location_deception"] > signal["r2"]["stuff_location"]


def test_the_r2_gain_interval_is_zero_for_identical_predictions_and_resamples_whole_pitchers():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(100), 2)
    y = rng.normal(size=200)
    pred = y * 0.3 + rng.normal(size=200)

    gain, lo, hi = ov.r2_gain_interval(y, pred, pred, groups, n_boot=100)

    assert gain == 0.0 and lo == 0.0 and hi == 0.0


def test_the_timing_rule_aligns_signs_and_demotes_only_when_both_outcomes_rule_out_a_real_effect():
    def coefs(timing_k, timing_x):
        base = {"whiff_index": {"coef": 0.0125, "lo": 0.0077, "hi": 0.0173}}
        return {"k_pct": {**base, "timing_index": timing_k}, "xwoba": {"whiff_index": {"coef": -0.0036, "lo": -0.0059, "hi": -0.0013}, "timing_index": timing_x}}

    flat_k = {"coef": 0.0, "lo": -0.001, "hi": 0.001}                         # upper end 8% of whiff's: under the 25% rule
    flat_x = {"coef": 0.0, "lo": -0.0005, "hi": 0.0005}                       # xwOBA: lower is better, so +0.0005 is a 14% effect: under the rule
    wide_x = {"coef": 0.0, "lo": -0.003, "hi": 0.003}                         # lo -0.003 means a 0.003 benefit, 83% of whiff's: not ruled out

    assert ov.timing_rule(coefs(flat_k, flat_x))["demote"] is True
    result = ov.timing_rule(coefs(flat_k, wide_x))
    assert result["demote"] is False and result["outcomes"]["xwoba"]["below_rule"] is False and result["outcomes"]["k_pct"]["below_rule"] is True


def test_stability_benchmarks_use_pairs_qualified_in_both_seasons_with_enough_batters_faced():
    ps, outcomes = synthetic(n=300, seed=5)
    ps.loc[(ps["pitcher"] < 30) & (ps["season"] == 2025), "qualified"] = False

    result = ov.stability_benchmarks(ps, outcomes)

    assert result["n"] == 270 and set(result["r"]) == set(ov.STABILITY_METRICS)
