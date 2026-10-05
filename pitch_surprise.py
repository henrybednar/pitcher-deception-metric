"""
Pitcher Deception Project: does an unpredictable pitch beat its expectation?
============================================================================
Previous-pitch features (speed change, location change, pitch-type change and so on) explain almost none
of the pitch-level residual, but they describe only the pitch before. The sequencing idea hitters care
about is predictability: how likely was this pitch type, given what the pitcher usually does in this
situation? This step measures that and tests it.

Pitch surprise. For each pitcher-season, a regularized multinomial logistic model predicts the pitch type
from the situation a hitter can see: the count, the batter's side, the previous two pitch types, how the
previous pitch ended, the outs and the baserunners. It is cross-fitted by game, so a pitch is always scored
by a model that never saw its game. Surprise is -log of the probability the model gave the pitch type that
was thrown (conditional surprise). The same measure from the pitcher's usage shares alone is the marginal
surprise, so marginal minus conditional is how much the situation told a hitter, the predictability gain.

Two tests.
  Pitch level   Within each pitcher x season x pitch type, do the pitches thrown when they were more
                surprising have a higher actual-minus-expected result? Comparing a pitcher's own pitches of
                one type removes who the pitcher is and which pitch it was; the residual already controls
                count, batter and location.
  Season level  Do pitchers with less predictable pitch choices score higher, beyond how widely they spread
                their pitch types (marginal surprise)? Optionally controlling for Stuff+.

Reads output/per_pitch_predictions.csv, output/pitcher_season.csv and the raw pitch file.
Writes output/pitch_surprise.json.
"""

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from export_site_stats import real_role_label
from raw_alignment import regular_season_rows
from reliability_and_ci import BINARY_OUTCOMES, COMPOSITE_OUTCOMES, CONTINUOUS_OUTCOMES

PREDICTIONS_FILE = "output/per_pitch_predictions.csv"
PITCHER_SEASON_FILE = "output/pitcher_season.csv"
RAW_FILE = "raw/statcast_pitch_level_2025_2026.csv"
OUT_FILE = "output/pitch_surprise.json"
SPECS = {**BINARY_OUTCOMES, **CONTINUOUS_OUTCOMES}
PITCH_ALIASES = {"SV": "ST", "FO": "FS"}
RESULT_OF = {
    "ball": "ball", "blocked_ball": "ball", "automatic_ball": "ball", "hit_by_pitch": "ball", "pitchout": "ball",
    "called_strike": "called", "automatic_strike": "called",
    "swinging_strike": "miss", "swinging_strike_blocked": "miss", "missed_bunt": "miss", "swinging_pitchout": "miss",
    "foul": "foul", "foul_tip": "foul", "foul_bunt": "foul", "bunt_foul_tip": "foul",
}
RAW_CONTEXT_COLS = ["game_pk", "at_bat_number", "pitch_number", "pitch_type", "description", "balls", "strikes", "stand",
                    "outs_when_up", "on_1b", "on_2b", "on_3b"]
MIN_PITCHES = 300                 # per pitcher-season, to fit a situation model at all
N_FOLDS = 5
REGULARIZATION = 0.3              # inverse L2 strength; the situation features are many and a pitcher's season is short
PROB_FLOOR = 1e-3                 # probability given to a pitch type the training games never showed
MIN_CELL_ROWS = 10
UNITS = {"pp": 100.0}


def previous_pitch_context(raw: pd.DataFrame) -> pd.DataFrame:
    """The previous two pitch types and how the previous pitch ended, within the same pitcher's plate
    appearance, indexed like `raw` ("NONE" where there was no previous pitch). Needs game_pk, at_bat_number,
    pitcher, pitch_number, pitch_type and description."""
    keys = ["game_pk", "at_bat_number", "pitcher"]
    ordered = raw.sort_values(keys + ["pitch_number"])[keys + ["pitch_type", "description"]].copy()
    ordered["kind"] = ordered["pitch_type"].replace(PITCH_ALIASES).fillna("OTHER")
    ordered["result"] = ordered["description"].map(RESULT_OF).fillna("other")
    grouped = ordered.groupby(keys, sort=False)
    context = pd.DataFrame({"prev1": grouped["kind"].shift(1), "prev2": grouped["kind"].shift(2),
                            "prev_result": grouped["result"].shift(1)}, index=ordered.index).fillna("NONE")
    return context.reindex(raw.index)


def situation_matrix(raw: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """One-hot situation features a hitter can see before the pitch."""
    frame = pd.DataFrame({
        "count": raw["balls"].fillna(0).astype(int).astype(str) + "-" + raw["strikes"].fillna(0).astype(int).astype(str),
        "stand": raw["stand"].astype(str),
        "prev1": context["prev1"], "prev2": context["prev2"], "prev_result": context["prev_result"],
        "outs": raw["outs_when_up"].fillna(0).astype(int).astype(str),
        "first": raw["on_1b"].notna().astype(int).astype(str), "second": raw["on_2b"].notna().astype(int).astype(str),
        "third": raw["on_3b"].notna().astype(int).astype(str),
    })
    return pd.get_dummies(frame, dtype=np.uint8)


def cross_fitted_surprisal(types: np.ndarray, X: np.ndarray, games: np.ndarray, n_folds: int = N_FOLDS
                           ) -> tuple[np.ndarray, np.ndarray]:
    """(conditional, marginal) surprise in nats for one pitcher-season, each pitch scored by a model fit on
    the other games. NaN throughout for a pitcher with one pitch type or fewer than two games."""
    types, games = np.asarray(types), np.asarray(games)
    conditional, marginal = np.full(len(types), np.nan), np.full(len(types), np.nan)
    classes = np.unique(types)
    folds = min(n_folds, len(np.unique(games)))
    if len(classes) < 2 or folds < 2:
        return conditional, marginal
    for train, test in GroupKFold(n_splits=folds).split(X, types, groups=games):
        seen, counts = np.unique(types[train], return_counts=True)
        share = dict(zip(seen, (counts + 1) / (counts.sum() + len(classes))))
        unseen_share = 1 / (counts.sum() + len(classes))
        marginal[test] = -np.log([share.get(t, unseen_share) for t in types[test]])
        if len(seen) < 2:
            conditional[test] = marginal[test]
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=REGULARIZATION, max_iter=300).fit(X[train], types[train])
        probability = model.predict_proba(X[test])
        column = {c: i for i, c in enumerate(model.classes_)}
        p = np.array([probability[row, column[t]] if t in column else PROB_FLOOR for row, t in enumerate(types[test])])
        conditional[test] = -np.log(np.clip(p, PROB_FLOOR, 1.0))
    return conditional, marginal


def pitch_surprisal(pitcher: np.ndarray, season: np.ndarray, game: np.ndarray, types: pd.Series, X: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """cross_fitted_surprisal for every pitcher-season with at least MIN_PITCHES scored pitches, by row."""
    frame = pd.DataFrame({"pitcher": pitcher, "season": season})
    scored = types.notna().to_numpy()
    groups = [idx[scored[idx]] for idx in frame.groupby(["pitcher", "season"]).indices.values()]
    groups = [idx for idx in groups if len(idx) >= MIN_PITCHES]
    results = Parallel(n_jobs=-1)(
        delayed(cross_fitted_surprisal)(types.to_numpy()[idx], X[idx].astype(np.float32), game[idx]) for idx in groups)
    conditional, marginal = np.full(len(types), np.nan), np.full(len(types), np.nan)
    for idx, (c, m) in zip(groups, results):
        conditional[idx], marginal[idx] = c, m
    return conditional, marginal


def within_cell_slope(frame: pd.DataFrame, y: str, x: str, cell: list[str], cluster: str,
                      controls: tuple[str, ...] = (), min_rows: int = MIN_CELL_ROWS) -> dict:
    """Slope of y on x after taking each cell's mean out of y, x and the controls, with a 95% interval from
    standard errors clustered on `cluster`. Cells with fewer than `min_rows` rows are dropped."""
    columns = [y, x, *controls]
    d = frame.dropna(subset=columns)
    d = d[d.groupby(cell)[y].transform("size") >= min_rows]
    centred = d[columns] - d.groupby(cell)[columns].transform("mean")
    model = sm.OLS(centred[y].to_numpy(float), centred[[x, *controls]].to_numpy(float))
    if d[cluster].nunique() < 2:                                  # one cluster carries no information about spread
        fit, se = model.fit(), float("nan")
        beta = float(fit.params[0])
    else:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
        beta, se = float(fit.params[0]), float(fit.bse[0])
    sd_x = float(centred[x].std())
    return {"beta": beta, "se": se, "lo": beta - 1.96 * se, "hi": beta + 1.96 * se, "sd_x": sd_x, "per_sd": beta * sd_x,
            "n": int(len(d)), "cells": int(d.groupby(cell).ngroups), "pitchers": int(d[cluster].nunique())}


def season_regression(table: pd.DataFrame, outcome: str, predictors: list[str], controls: list[str] | None = None) -> dict:
    """OLS of the outcome on standardized predictors (and controls), errors clustered on pitcher. Each
    predictor's effect is reported in outcome units per +1 SD of that predictor."""
    columns = [outcome, *predictors, *(controls or [])]
    d = table.dropna(subset=columns)
    Z = (d[predictors + (controls or [])] - d[predictors + (controls or [])].mean()) / d[predictors + (controls or [])].std()
    fit = sm.OLS(d[outcome].to_numpy(float), sm.add_constant(Z.to_numpy(float))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["pitcher"])[0]})
    out = {}
    for i, name in enumerate(predictors + (controls or []), start=1):
        b, se = float(fit.params[i]), float(fit.bse[i])
        out[name] = {"per_sd": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se}
    out["n"] = int(len(d))
    return out


def role_dummies(ps: pd.DataFrame) -> pd.DataFrame:
    """is_rp and is_mr (starters are the base) from the real games-started share; NaN where it is unknown."""
    share = (ps["games_started"] / ps["games"]).where(ps["games"] > 0)
    role = share.map(real_role_label)
    known = role.notna()
    return pd.DataFrame({"is_rp": (role == "RP").astype(float).where(known), "is_mr": (role == "MR").astype(float).where(known)},
                        index=ps.index)


def pitch_level_tests(pred: pd.DataFrame) -> dict:
    """Within-cell slope of each member's residual on conditional surprise, without and with a control for
    whether the pitch type changed from the previous pitch."""
    out = {}
    for label in COMPOSITE_OUTCOMES:
        spec, expected = SPECS[label], f"{label}_expected_full"
        rows = spec["subset"](pred) & pred[expected].notna() & pred["cond"].notna()
        scale = UNITS["pp"] if label in BINARY_OUTCOMES else 1.0
        frame = pred.loc[rows, ["pitcher", "season", "pitch_type", "cond", "changed"]].copy()
        frame["resid"] = (pred.loc[rows, spec["target"]].astype(float) - pred.loc[rows, expected]) * scale
        out[label] = {"unit": "pp" if label in BINARY_OUTCOMES else "raw",
                      "surprise": within_cell_slope(frame, "resid", "cond", ["pitcher", "season", "pitch_type"], "pitcher"),
                      "surprise_given_change": within_cell_slope(frame, "resid", "cond", ["pitcher", "season", "pitch_type"],
                                                                 "pitcher", controls=("changed",))}
    return out


def season_level_tests(pred: pd.DataFrame, ps: pd.DataFrame) -> dict:
    means = (pred[pred["cond"].notna()].groupby(["pitcher", "season"]).agg(cond=("cond", "mean"), marg=("marg", "mean"), n=("cond", "size"))
             .reset_index())
    means["gain"] = means["marg"] - means["cond"]
    table = ps.join(role_dummies(ps))
    table = table[table["qualified"]].merge(means, on=["pitcher", "season"], how="inner")
    out = {"pitcher_seasons": int(len(table)), "mean_conditional": float(means["cond"].mean()),
           "mean_marginal": float(means["marg"].mean()), "mean_gain": float(means["gain"].mean()),
           "share_with_gain": float((means["gain"] > 0).mean()), "sd_gain": float(means["gain"].std())}
    for outcome in [f"{m}_index" for m in COMPOSITE_OUTCOMES] + ["deception_plus"]:
        out[outcome] = {"marginal_and_gain": season_regression(table, outcome, ["marg", "gain"]),
                        "with_stuff_plus": season_regression(table, outcome, ["marg", "gain"], ["stuff_plus"]),
                        "with_stuff_plus_and_role": season_regression(table, outcome, ["marg", "gain"], ["stuff_plus", "is_rp", "is_mr"])}
    return out


if __name__ == "__main__":
    print("loading per-pitch predictions and the raw situation columns...", flush=True)
    pred = pd.read_csv(PREDICTIONS_FILE)
    raw = regular_season_rows(RAW_FILE, RAW_CONTEXT_COLS, pred)
    raw["pitcher"] = pred["pitcher"].to_numpy()
    context = previous_pitch_context(raw)
    X = situation_matrix(raw, context).to_numpy()
    print(f"situation features: {X.shape[1]}; fitting one model per pitcher-season...", flush=True)
    pred["cond"], pred["marg"] = pitch_surprisal(pred["pitcher"].to_numpy(), pred["season"].to_numpy(), raw["game_pk"].to_numpy(),
                                                 pred["pitch_type"], X)
    pred["changed"] = np.where(context["prev1"] == "NONE", np.nan, (context["prev1"] != pred["pitch_type"]).astype(float))
    modeled = pred["cond"].notna()
    print(f"scored pitches with a surprise value: {int(modeled.sum()):,} of {int(pred['pitch_type'].notna().sum()):,}", flush=True)

    ps = pd.read_csv(PITCHER_SEASON_FILE)
    result = {"pitch_level": pitch_level_tests(pred), "season_level": season_level_tests(pred, ps)}
    s = result["season_level"]
    print(f"\nmean surprise {s['mean_conditional']:.3f} nats given the situation, {s['mean_marginal']:.3f} from usage alone; "
          f"gain {s['mean_gain']:.3f} (SD {s['sd_gain']:.3f}); {s['share_with_gain']:.0%} of pitcher-seasons gain", flush=True)
    print("\npitch level: change in actual-minus-expected result per +1 nat of surprise, within pitcher x season x pitch type", flush=True)
    for label, v in result["pitch_level"].items():
        a, b = v["surprise"], v["surprise_given_change"]
        print(f"  {label:10s} {a['beta']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] {v['unit']} (per within-cell SD of {a['sd_x']:.2f} nats: {a['per_sd']:+.3f})   "
              f"given pitch-type change {b['beta']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]   n={a['n']:,}", flush=True)
    print("\nseason level: index points per +1 SD (qualified pitcher-seasons)", flush=True)
    for outcome in [f"{m}_index" for m in COMPOSITE_OUTCOMES] + ["deception_plus"]:
        for name in ("marginal_and_gain", "with_stuff_plus", "with_stuff_plus_and_role"):
            r = s[outcome][name]
            print(f"  {outcome:16s} {name:25s} usage spread {r['marg']['per_sd']:+.2f} [{r['marg']['lo']:+.2f}, {r['marg']['hi']:+.2f}]   "
                  f"predictability gain {r['gain']['per_sd']:+.2f} [{r['gain']['lo']:+.2f}, {r['gain']['hi']:+.2f}]   n={r['n']}", flush=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUT_FILE}. Done.", flush=True)
