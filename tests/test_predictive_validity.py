import numpy as np
import pandas as pd
import pytest

import predictive_validity as pv


def forecast_frame(n=400, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    expected = rng.normal(0.25, 0.03, n)
    prior = rng.normal(0, 1, n)
    y = expected + (0.01 * prior if informative else 0.0) + rng.normal(0, 0.01, n)
    return pd.DataFrame({"pitcher": np.arange(n), "exp": expected, "prior": prior, "y": y})


def test_an_informative_prior_season_score_passes_the_test_and_adds_cross_validated_r2():
    result = pv.evaluate(forecast_frame(), "exp", "prior", "y")

    assert result["n"] == 400 and result["pitchers"] == 400
    assert result["p_value"] < 0.001
    assert result["cv_delta_r2"] > 0.02


def test_a_pure_noise_prior_season_score_does_not_pass_and_cannot_improve_cross_validated_r2():
    result = pv.evaluate(forecast_frame(informative=False), "exp", "prior", "y")

    assert result["p_value"] > 0.01
    assert result["cv_delta_r2"] < 0.005
    assert result["delta_r2_in_sample"] >= 0        # in-sample R2 can only go up, which is why the test and CV exist


def test_the_added_term_p_value_agrees_with_the_ols_t_test_when_every_row_is_its_own_cluster():
    data = forecast_frame(n=200, seed=3)
    X, y = data[["exp", "prior"]].to_numpy(), data["y"].to_numpy()

    clustered = pv.added_term_p_value(X, y, data["pitcher"].to_numpy())

    design = np.column_stack([np.ones(200), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    # with one row per cluster the cluster-robust variance is the HC0 sandwich
    bread = np.linalg.inv(design.T @ design)
    cov = bread @ (design.T * resid ** 2) @ design @ bread
    z = beta[-1] / np.sqrt(cov[-1, -1])
    from scipy import stats
    assert clustered == pytest.approx(2 * (1 - stats.norm.cdf(abs(z))), rel=0.05)


def test_clustering_widens_the_test_when_a_pitchers_rows_repeat_the_same_noise():
    rng = np.random.default_rng(5)
    pitchers = np.repeat(np.arange(100), 2)
    shared = np.repeat(rng.normal(0, 0.02, 100), 2)                  # a pitcher's two rows share their noise
    exp = rng.normal(0.25, 0.03, 200)
    prior = np.repeat(rng.normal(0, 1, 100), 2)
    y = exp + shared + 0.002 * prior
    X = np.column_stack([exp, prior])

    clustered = pv.added_term_p_value(X, y, pitchers)
    independent = pv.added_term_p_value(X, y, np.arange(200))

    assert clustered > independent


def test_next_season_frame_pairs_each_season_with_the_previous_one_and_drops_pitchers_without_one():
    agg = pd.DataFrame({"pitcher": [1, 1, 1, 2, 3], "season": [2024, 2025, 2026, 2025, 2026],
                        "actual_mean": [0.1, 0.2, 0.3, 0.4, 0.5], "expected_mean": [0.11, 0.21, 0.31, 0.41, 0.51]})
    ps = pd.DataFrame({"pitcher": [1, 1, 1, 2, 3], "season": [2024, 2025, 2026, 2025, 2026],
                       "whiff_index": [90.0, 100.0, 110.0, 120.0, 130.0]})

    out = pv.next_season_frame(agg, ps, "whiff_index").sort_values(["pitcher", "season"])

    assert out[["pitcher", "season"]].values.tolist() == [[1, 2025], [1, 2026]]       # 2 has no 2024 row, 3 no 2025 row
    assert out["prior_index"].tolist() == [90.0, 100.0]
    assert out["y_next"].tolist() == [0.2, 0.3]
