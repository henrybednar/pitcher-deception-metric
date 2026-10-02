import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from driver_features import fdr_adjusted_p_values


def noisy_frame(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    return X, rng


def test_fdr_adjusted_p_values_flag_the_real_effect_and_not_the_noise_columns():
    X, rng = noisy_frame()
    y = 0.5 * X["f0"] + rng.normal(size=len(X))

    q = fdr_adjusted_p_values(X, y)

    assert q["f0"] < 0.001
    assert (q.drop("f0") > 0.05).all()


def test_fdr_adjusted_p_values_are_never_below_the_raw_p_values():
    X, rng = noisy_frame(seed=1)
    y = 0.1 * X["f1"] + rng.normal(size=len(X))

    q = fdr_adjusted_p_values(X, y)
    raw = sm.OLS(y, sm.add_constant(X)).fit().pvalues[X.columns]

    assert (q >= raw - 1e-12).all()
    assert list(q.index) == list(X.columns)


def test_fdr_adjusted_p_values_do_not_depend_on_the_scale_of_a_feature():
    X, rng = noisy_frame(seed=2)
    y = 0.3 * X["f2"] + rng.normal(size=len(X))
    rescaled = X.assign(f2=X["f2"] * 1000)

    assert fdr_adjusted_p_values(rescaled, y)["f2"] == pytest.approx(fdr_adjusted_p_values(X, y)["f2"])
