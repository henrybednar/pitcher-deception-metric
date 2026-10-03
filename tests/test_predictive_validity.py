import numpy as np
import pandas as pd
import pytest
from scipy import stats

import predictive_validity as pv


def forecast_frame(n=400, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    expected = rng.normal(0.25, 0.03, n)
    prior = rng.normal(0, 1, n)
    y = expected + (0.01 * prior if informative else 0.0) + rng.normal(0, 0.01, n)
    return pd.DataFrame({"exp": expected, "prior": prior, "y": y})


def test_an_informative_prior_season_score_passes_the_nested_f_test_and_adds_cross_validated_r2():
    result = pv.evaluate(forecast_frame(), "exp", "prior", "y")

    assert result["n"] == 400
    assert result["f_pvalue"] < 0.001
    assert result["cv_delta_r2"] > 0.02


def test_a_pure_noise_prior_season_score_does_not_pass_and_cannot_improve_cross_validated_r2():
    result = pv.evaluate(forecast_frame(informative=False), "exp", "prior", "y")

    assert result["f_pvalue"] > 0.01
    assert result["cv_delta_r2"] < 0.005
    assert result["delta_r2_in_sample"] >= 0        # in-sample R2 can only go up, which is why the F test and CV exist


def test_the_nested_f_test_agrees_with_the_closed_form_for_one_added_predictor():
    data = forecast_frame(n=200, seed=3)
    X_a, X_b, y = data[["exp"]].to_numpy(), data[["exp", "prior"]].to_numpy(), data["y"].to_numpy()

    p = pv.nested_f_test(X_a, X_b, y)

    rss_a = np.sum((y - np.polyval(np.polyfit(X_a[:, 0], y, 1), X_a[:, 0])) ** 2)
    design = np.column_stack([np.ones(200), X_b])
    rss_b = np.sum((y - design @ np.linalg.lstsq(design, y, rcond=None)[0]) ** 2)
    f_stat = (rss_a - rss_b) / (rss_b / (200 - 3))
    assert p == pytest.approx(1 - stats.f.cdf(f_stat, 1, 200 - 3))
