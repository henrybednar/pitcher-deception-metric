"""
Pitcher Deception Project — predictive validity check
========================================================
Does a pitcher's 2025 deception score predict their 2026 ACTUAL outcome
rate, on top of what 2026's own stuff+location+opponent expectation
already predicts?

  Model A: 2026_actual_rate ~ 2026_own_expected_full
  Model B: 2026_actual_rate ~ 2026_own_expected_full + 2025_{label}_index

Two numbers reported per component, not just one:
  - `f_pvalue`: a nested-model partial-F test on the added coefficient —
    the correct significance test, since in-sample R² is mathematically
    guaranteed to be >= when adding ANY second predictor (even pure noise),
    so a raw in-sample delta-R^2 alone cannot distinguish real signal from
    that guaranteed-non-negative artifact.
  - `cv_delta_r2`: 5-fold cross-validated (out-of-sample) R^2 delta, which
    is NOT guaranteed non-negative — a genuinely uninformative predictor
    will show a negative or near-zero out-of-sample delta on average.

2026 is a partial season, so treat this as a direction-of-effect check on
a real held-out year, not a precise long-run estimate.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from reliability_and_ci import (
    BINARY_OUTCOMES,
    CONTINUOUS_OUTCOMES,
    MIN_N_FOR_SCORE,
    game_level_table,
    pitcher_season_point_estimate,
)

OUTCOME_SPECS = {**BINARY_OUTCOMES, **CONTINUOUS_OUTCOMES}
CV = KFold(n_splits=5, shuffle=True, random_state=42)


def in_sample_r2(X: np.ndarray, y: np.ndarray) -> float:
    return LinearRegression().fit(X, y).score(X, y)


def cv_r2(X: np.ndarray, y: np.ndarray) -> float:
    return cross_val_score(LinearRegression(), X, y, cv=CV, scoring="r2").mean()


def nested_f_test(X_a: np.ndarray, X_b: np.ndarray, y: np.ndarray) -> float:
    """Partial-F test: is the extra predictor in X_b (vs. X_a) significant?
    X_b must be X_a with exactly one additional column appended."""
    n = len(y)
    rss_a = np.sum((y - LinearRegression().fit(X_a, y).predict(X_a)) ** 2)
    rss_b = np.sum((y - LinearRegression().fit(X_b, y).predict(X_b)) ** 2)
    df_b = n - X_b.shape[1] - 1
    f_stat = ((rss_a - rss_b) / 1) / (rss_b / df_b)
    return 1 - stats.f.cdf(max(f_stat, 0), 1, df_b)


def evaluate(data: pd.DataFrame, x_col: str, prior_col: str, y_col: str) -> dict:
    X_a = data[[x_col]].values
    X_b = data[[x_col, prior_col]].values
    y = data[y_col].values
    r2_a_in, r2_b_in = in_sample_r2(X_a, y), in_sample_r2(X_b, y)
    return {
        "n": len(data),
        "r2_stuff_only": round(r2_a_in, 4),
        "r2_plus_2025_deception": round(r2_b_in, 4),
        "delta_r2_in_sample": round(r2_b_in - r2_a_in, 4),
        "cv_delta_r2": round(cv_r2(X_b, y) - cv_r2(X_a, y), 4),
        "f_pvalue": round(nested_f_test(X_a, X_b, y), 6),
    }


if __name__ == "__main__":
    df = pd.read_csv("per_pitch_predictions.csv")
    ps = pd.read_csv("pitcher_season.csv")

    rows = []
    for label, spec in OUTCOME_SPECS.items():
        mask = spec["subset"](df) & df[f"{label}_expected_full"].notna() & (df["season"] == 2026)
        games_26 = game_level_table(df[mask], label, spec["target"], baseline="full")
        agg_26 = pitcher_season_point_estimate(games_26)
        agg_26 = agg_26[agg_26["n"] >= MIN_N_FOR_SCORE][["pitcher", "actual_mean", "expected_mean"]]
        agg_26 = agg_26.rename(columns={"actual_mean": "y_2026", "expected_mean": "exp_2026_full"})

        prior = ps.loc[ps["season"] == 2025, ["pitcher", f"{label}_index"]].rename(
            columns={f"{label}_index": "prior_index"})

        data = agg_26.merge(prior, on="pitcher", how="inner").dropna()
        if len(data) < 30:
            print(f"{label}: n={len(data)} too small, skipping")
            continue

        result = {"label": label, **evaluate(data, "exp_2026_full", "prior_index", "y_2026")}
        rows.append(result)
        print(f"{label}: n={result['n']}, R2(2026 stuff alone)={result['r2_stuff_only']:.4f}, "
              f"R2(+2025 index)={result['r2_plus_2025_deception']:.4f}, "
              f"in-sample delta={result['delta_r2_in_sample']:+.4f}, "
              f"CV delta={result['cv_delta_r2']:+.4f}, p={result['f_pvalue']:.4g}", flush=True)

    # composite-level check: does 2025 Deception+ predict 2026 actual whiff rate?
    spec = BINARY_OUTCOMES["whiff"]
    mask = spec["subset"](df) & df["whiff_expected_full"].notna()
    sub = df[mask]
    games_26 = game_level_table(sub[sub["season"] == 2026], "whiff", spec["target"], baseline="full")
    agg_26 = pitcher_season_point_estimate(games_26)
    agg_26 = agg_26[agg_26["n"] >= MIN_N_FOR_SCORE][["pitcher", "actual_mean", "expected_mean"]]
    agg_26 = agg_26.rename(columns={"actual_mean": "y_2026", "expected_mean": "exp_2026_full"})
    prior = ps.loc[ps["season"] == 2025, ["pitcher", "deception_plus"]]
    data = agg_26.merge(prior, on="pitcher", how="inner").dropna()
    result = {"label": "composite_to_whiff", **evaluate(data, "exp_2026_full", "deception_plus", "y_2026")}
    rows.append(result)
    print(f"composite_to_whiff: n={result['n']}, R2(2026 stuff alone)={result['r2_stuff_only']:.4f}, "
          f"R2(+2025 Deception+)={result['r2_plus_2025_deception']:.4f}, "
          f"in-sample delta={result['delta_r2_in_sample']:+.4f}, "
          f"CV delta={result['cv_delta_r2']:+.4f}, p={result['f_pvalue']:.4g}", flush=True)

    pd.DataFrame(rows).to_csv("predictive_validity_report.csv", index=False)
    print("\nSaved predictive_validity_report.csv. Done.")
