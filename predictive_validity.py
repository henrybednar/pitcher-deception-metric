"""
Pitcher Deception Project — predictive validity check
========================================================
Does a pitcher's deception score in one season predict their ACTUAL outcome rate the next season, on top
of what that next season's own stuff+location+opponent expectation already predicts?

  Model A: next_actual_rate ~ next_own_expected_full
  Model B: next_actual_rate ~ next_own_expected_full + this_season_{label}_index

Every pair of consecutive seasons counts (2024 to 2025 and 2025 to 2026), stacked into one regression, so a
pitcher who threw all three seasons contributes two rows.

Two numbers reported per component, not just one:
  - `p_value`: a test of the added coefficient with standard errors clustered on pitcher, since a pitcher's two
    rows share a season. In-sample R² is mathematically guaranteed to be >= when adding ANY second predictor
    (even pure noise), so a raw in-sample delta-R^2 alone cannot distinguish real signal from that
    guaranteed-non-negative artifact, and the test is the one that can.
  - `cv_delta_r2`: 5-fold cross-validated (out-of-sample) R^2 delta with folds grouped by pitcher, which is NOT
    guaranteed non-negative — a genuinely uninformative predictor will show a negative or near-zero
    out-of-sample delta on average.

Two held-out year-ahead pairs are still a short record; treat this as a direction-of-effect check, not a
precise long-run estimate.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from reliability_and_ci import (
    BINARY_OUTCOMES,
    CONTINUOUS_OUTCOMES,
    MIN_N_FOR_SCORE,
    game_level_table,
    pitcher_season_point_estimate,
)

OUTCOME_SPECS = {**BINARY_OUTCOMES, **CONTINUOUS_OUTCOMES}
CV_SPLITS = 5
CV_SEED = 42
MIN_ROWS = 30


def in_sample_r2(X: np.ndarray, y: np.ndarray) -> float:
    return LinearRegression().fit(X, y).score(X, y)


def cv_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    cv = GroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    return cross_val_score(LinearRegression(), X, y, cv=cv, groups=groups, scoring="r2").mean()


def added_term_p_value(X_b: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """p-value of the last column of X_b in an OLS fit, with standard errors clustered on `groups`."""
    fit = sm.OLS(y, sm.add_constant(X_b)).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return float(fit.pvalues[-1])


def next_season_frame(agg: pd.DataFrame, ps: pd.DataFrame, prior_col: str) -> pd.DataFrame:
    """One row per pitcher and season that has a scored previous season: the season's own actual and expected rate
    (agg: pitcher, season, actual_mean, expected_mean) beside the previous season's `prior_col` value from ps."""
    prior = (ps[["pitcher", "season", prior_col]].assign(season=lambda d: d["season"] + 1)
             .rename(columns={prior_col: "prior_index"}))
    out = agg[["pitcher", "season", "actual_mean", "expected_mean"]].merge(prior, on=["pitcher", "season"])
    return out.rename(columns={"actual_mean": "y_next", "expected_mean": "exp_next_full"}).dropna().reset_index(drop=True)


def evaluate(data: pd.DataFrame, x_col: str, prior_col: str, y_col: str, group_col: str = "pitcher") -> dict:
    X_a = data[[x_col]].values
    X_b = data[[x_col, prior_col]].values
    y = data[y_col].values
    groups = data[group_col].values
    r2_a_in, r2_b_in = in_sample_r2(X_a, y), in_sample_r2(X_b, y)
    return {
        "n": len(data),
        "pitchers": int(data[group_col].nunique()),
        "r2_stuff_only": round(r2_a_in, 4),
        "r2_plus_prior_deception": round(r2_b_in, 4),
        "delta_r2_in_sample": round(r2_b_in - r2_a_in, 4),
        "cv_delta_r2": round(cv_r2(X_b, y, groups) - cv_r2(X_a, y, groups), 4),
        "p_value": round(added_term_p_value(X_b, y, groups), 6),
    }


def pitcher_season_rates(df: pd.DataFrame, label: str, spec: dict) -> pd.DataFrame:
    """Actual and expected mean of one outcome for every scored pitcher-season."""
    mask = spec["subset"](df) & df[f"{label}_expected_full"].notna()
    agg = pitcher_season_point_estimate(game_level_table(df[mask], label, spec["target"], baseline="full"))
    return agg[agg["n"] >= MIN_N_FOR_SCORE]


def describe(result: dict) -> str:
    return (f"n={result['n']} ({result['pitchers']} pitchers), R2(next season's expectation alone)={result['r2_stuff_only']:.4f}, "
            f"R2(+prior score)={result['r2_plus_prior_deception']:.4f}, in-sample delta={result['delta_r2_in_sample']:+.4f}, "
            f"CV delta={result['cv_delta_r2']:+.4f}, p={result['p_value']:.4g}")


if __name__ == "__main__":
    df = pd.read_csv("output/per_pitch_predictions.csv")
    ps = pd.read_csv("output/pitcher_season.csv")

    rows = []
    for label, spec in OUTCOME_SPECS.items():
        data = next_season_frame(pitcher_season_rates(df, label, spec), ps, f"{label}_index")
        if len(data) < MIN_ROWS:
            print(f"{label}: n={len(data)} too small, skipping")
            continue
        result = {"label": label, **evaluate(data, "exp_next_full", "prior_index", "y_next")}
        rows.append(result)
        print(f"{label}: {describe(result)}", flush=True)

    # composite-level check: does a season's Deception+ predict the next season's actual whiff rate?
    data = next_season_frame(pitcher_season_rates(df, "whiff", BINARY_OUTCOMES["whiff"]), ps, "deception_plus")
    result = {"label": "composite_to_whiff", **evaluate(data, "exp_next_full", "prior_index", "y_next")}
    rows.append(result)
    print(f"composite_to_whiff: {describe(result)}", flush=True)

    pd.DataFrame(rows).to_csv("output/predictive_validity_report.csv", index=False)
    print("\nSaved predictive_validity_report.csv. Done.")
