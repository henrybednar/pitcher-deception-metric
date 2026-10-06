"""
Pitcher Deception Project: are the shrinkage intervals calibrated out of sample?
================================================================================
Each pitcher-season's games split into two halves (game_pk parity). The posterior built from one half
(shrunk with the half's own sampling variance and design effect, with the between-pitcher variance
re-estimated from the half and repeated pitchers collapsed as in reliability_and_ci.py; the heterogeneity
test is skipped) is used to predict the other half's raw result. If the intervals are right, the
standardized prediction errors
    z = (other half's difference - posterior mean) / sqrt(posterior variance + other half's sampling variance)
have a standard deviation of 1 and fall within 1.96 for 95% of pitcher-seasons, at every sample size.
A model that understates the sampling noise, or the game-level clustering, shows up as an SD above 1 where
samples are largest.

What it does and does not show. It tests the measurement noise within a season, which the design effect and
the shrinkage model. Anything the two halves share (park, opponents, a persistent model error, true talent
itself) cancels, so it says nothing about uncertainty in skill, and the design-effect curve is estimated from
the same games, so the check is partly self-consistent: it flagged the pooled factor, and the cluster-size
curve was built in response. A simulation with known talent (a one-off, in the README) covers true talent
independently.

Run after add_timing_direction.py. Reads per_pitch_predictions.csv, pitcher_season.csv (who qualified) and
design_effects.json. Writes interval_coverage.json for whiff, chase and weak contact (the outcomes whose
per-pitch variance is a Bernoulli variance computed in half_tables; ground ball and called strike are binary
too but are not members, and timing and whiff miss would need their residual variances rebuilt here).
"""

import json

import numpy as np
import pandas as pd

import reliability_and_ci as rc

LABELS = ("whiff", "chase", "weak")
MIN_PITCHES_PER_HALF = rc.MIN_N_FOR_SCORE
MIN_PITCHER_SEASONS = 30
PREDICTION_COLUMNS = ["pitcher", "season", "game_pk", "half", "is_swing", "is_in_zone", "is_bip", "is_whiff", "is_weak",
                      "whiff_expected_full", "chase_expected_full", "weak_expected_full"]


def half_tables(pp: pd.DataFrame, label: str) -> pd.DataFrame:
    """One row per pitcher-season with, for each half of its games: pitches, mean actual minus expected, the analytic
    sampling variance of that mean (no design effect yet) and the number of games."""
    spec = rc.BINARY_OUTCOMES[label]
    sub = pp[spec["subset"](pp) & pp[f"{label}_expected_full"].notna()]
    resid = sub[spec["target"]].astype(float) - sub[f"{label}_expected_full"]
    frame = pd.DataFrame({"pitcher": sub["pitcher"], "season": sub["season"], "game_pk": sub["game_pk"], "half": sub["half"],
                          "resid": resid, "var": rc.binary_pitch_variance(sub[f"{label}_expected_full"])})
    g = frame.groupby(["pitcher", "season", "half"]).agg(n=("resid", "size"), diff=("resid", "mean"), var_sum=("var", "sum"),
                                                          n_games=("game_pk", "nunique")).reset_index()
    g["sampling_var"] = g["var_sum"] / g["n"] ** 2
    wide = g.pivot(index=["pitcher", "season"], columns="half", values=["n", "diff", "sampling_var", "n_games"])
    wide.columns = [f"{name}{half}" for name, half in wide.columns]
    return wide.dropna().reset_index()


def predictive_z(wide: pd.DataFrame, curve: list[tuple[float, float]]) -> pd.DataFrame:
    """Standardized errors of predicting each half from the other, in both directions, with the smaller half's sample size."""
    out = []
    for source, target in ((0, 1), (1, 0)):
        effect = {h: rc.design_effect_for(pd.DataFrame({"n": wide[f"n{h}"], "n_games": wide[f"n_games{h}"]}), curve) for h in (source, target)}
        source_var = wide[f"sampling_var{source}"] * effect[source]
        target_var = wide[f"sampling_var{target}"] * effect[target]
        collapsed_diff, collapsed_var = rc.collapse_repeated_pitchers(wide["pitcher"].to_numpy(), wide[f"n{source}"].to_numpy(),
                                                                       wide[f"diff{source}"].to_numpy(), source_var.to_numpy())
        tau2 = rc.estimate_true_var(collapsed_diff, collapsed_var)
        center = float(np.average(wide[f"diff{source}"], weights=1 / (tau2 + source_var)))
        shrink = tau2 / (tau2 + source_var)
        posterior_mean = center + shrink * (wide[f"diff{source}"] - center)
        posterior_var = shrink * source_var
        z = (wide[f"diff{target}"] - posterior_mean) / np.sqrt(posterior_var + target_var)
        out.append(pd.DataFrame({"z": z, "n_min": np.minimum(wide["n0"], wide["n1"]),
                                 "m_min": np.minimum(wide["n0"] / wide["n_games0"], wide["n1"] / wide["n_games1"])}))
    return pd.concat(out, ignore_index=True)


def summarize(z: pd.DataFrame) -> dict:
    """Spread and coverage of the standardized errors, overall, in thirds of the smaller half's sample size and (when the
    scored pitches per game are given) in thirds of that cluster size, which is what the design effect depends on."""
    def by_third(column: str) -> list[float]:
        third = pd.qcut(z[column].rank(method="first"), 3, labels=False)
        return [float(z.loc[third == t, "z"].std()) for t in range(3)]

    summary = {"n": int(len(z) // 2), "z_mean": float(z["z"].mean()), "z_sd": float(z["z"].std()), "within_95": float((z["z"].abs() < 1.96).mean()),
               "within_80": float((z["z"].abs() < 1.2816).mean()), "z_sd_by_sample_third": by_third("n_min")}
    if "m_min" in z:
        summary["z_sd_by_cluster_third"] = by_third("m_min")
    return summary


def coverage(pp: pd.DataFrame, qualified: pd.DataFrame, curves: dict) -> dict:
    """The summary for every label, over qualified pitcher-seasons with enough pitches in both halves."""
    result = {}
    for label in LABELS:
        wide = half_tables(pp, label).merge(qualified, on=["pitcher", "season"])
        wide = wide[wide["qualified"] & (wide["n0"] >= MIN_PITCHES_PER_HALF) & (wide["n1"] >= MIN_PITCHES_PER_HALF)]
        if len(wide) < MIN_PITCHER_SEASONS:
            raise ValueError(f"{label}: only {len(wide)} qualified pitcher-seasons have {MIN_PITCHES_PER_HALF}+ pitches in both halves "
                             f"(need {MIN_PITCHER_SEASONS}); there is nothing to check coverage on")
        result[label] = summarize(predictive_z(wide, [tuple(p) for p in curves[label]]))
    return result


if __name__ == "__main__":
    print("loading per-pitch predictions...", flush=True)
    predictions = pd.read_csv("output/per_pitch_predictions.csv", usecols=PREDICTION_COLUMNS)
    qualified = pd.read_csv("output/pitcher_season.csv", usecols=["pitcher", "season", "qualified"])
    with open("output/design_effects.json", encoding="utf-8") as f:
        design_effects = json.load(f)
    result = coverage(predictions, qualified, design_effects)
    with open("output/interval_coverage.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for label, r in result.items():
        print(f"{label:6s} n={r['n']:,}  z SD {r['z_sd']:.3f}  within 1.96: {r['within_95']:.1%}  within 1.28: {r['within_80']:.1%}  "
              f"SD by sample third {[round(x, 2) for x in r['z_sd_by_sample_third']]}")
    print("Saved interval_coverage.json. Done.")
