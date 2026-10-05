"""
Pitcher Deception Project: does Deception+ predict next-season results?
=======================================================================
Every other check in this project validates Deception+ against its own components. This one asks the baseball
question: does a pitcher's score in one season help predict their results the next season, beyond what Stuff+ and
Location+ already say and beyond last season's own results?

Outcomes (regular season, from Statcast): strikeout rate, walk rate, xwOBA allowed (Savant's expected wOBA, with the
actual wOBA value on walks, strikeouts and the like), wOBA allowed, and run value per 100 pitches. For every pair of
consecutive seasons, the earlier season's scores (Stuff+, Location+, Deception+, the member indexes) are set against
the next season's outcome, for pitchers with at least MIN_NEXT_PA batters faced in the next season.

Predictor sets are compared by cross-validated R2 (folds grouped by pitcher, five random splits averaged), and the
gain from adding Deception+ comes with a 95% interval from resampling pitchers. A second view regresses each outcome
on the standardized member indexes (standard errors clustered on pitcher), which says which members carry it.

Timing check. Timing is the most reliable member but the one with the weakest case for mattering on the field. The
rule written down for it: demote it from the composite if, on both strikeout rate and xwOBA, the upper end of its
95% interval is under TIMING_RULE_SHARE of whiff's coefficient (signs aligned so that positive is better for the
pitcher). The result of the rule is printed and stored every run; it is not applied automatically.

Run after reliability_and_ci.py. Writes output/outcome_validation.json, which export_site_stats.py and
build_artifact.py read.
"""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from season_pairs import consecutive_pairs

RAW_FILE = "raw/statcast_pitch_level_2024_2026.csv"
RAW_COLUMNS = ["pitcher", "season", "game_type", "events", "description", "woba_value", "woba_denom",
               "estimated_woba_using_speedangle", "delta_pitcher_run_exp"]
CHUNKSIZE = 400_000
MIN_NEXT_PA = 150
CV_SPLITS = 5
CV_SEEDS = tuple(range(8))
N_BOOT = 1000
TIMING_RULE_SHARE = 0.25

STRIKEOUTS = {"strikeout", "strikeout_double_play"}
WALKS = {"walk"}
SWINGS = {"foul", "foul_tip", "hit_into_play", "swinging_strike", "swinging_strike_blocked"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked"}

# outcome column -> (label, better direction for the pitcher: +1 if higher is better, -1 if lower is better)
OUTCOMES = {
    "k_pct": ("strikeout rate", +1),
    "bb_pct": ("walk rate", -1),
    "xwoba": ("xwOBA allowed", -1),
    "woba": ("wOBA allowed", -1),
    "rv100": ("run value per 100 pitches", +1),
}
MEMBERS = ["whiff", "chase", "weak", "timing", "whiffmiss"]
STABILITY_METRICS = {"stuff_plus": "Stuff+", "location_plus": "Location+", "deception_plus": "Deception+", "whiff_rate": "whiff rate",
                     "k_pct": "strikeout rate", "bb_pct": "walk rate", "xwoba": "xwOBA allowed", "woba": "wOBA allowed", "rv100": "run value per 100 pitches"}


def chunk_totals(chunk: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher-season sums of one chunk of regular-season pitches: plate appearances, strikeouts, walks, the wOBA
    and xwOBA numerators and denominator, swings and whiffs, run value and pitches."""
    c = chunk[chunk["game_type"] == "R"].copy()
    ended = c["events"].notna()
    c["pa"] = ended.astype(int)
    c["k"] = c["events"].isin(STRIKEOUTS).astype(int)
    c["bb"] = c["events"].isin(WALKS).astype(int)
    c["woba_num"] = c["woba_value"].fillna(0.0)
    c["woba_den"] = c["woba_denom"].fillna(0.0)
    c["xwoba_num"] = np.where(ended, c["estimated_woba_using_speedangle"].fillna(c["woba_value"]).fillna(0.0), 0.0)
    c["swing"] = c["description"].isin(SWINGS).astype(int)
    c["whiff"] = c["description"].isin(WHIFFS).astype(int)
    c["rv"] = c["delta_pitcher_run_exp"].fillna(0.0)
    c["pitches"] = 1
    return c.groupby(["pitcher", "season"])[["pa", "k", "bb", "woba_num", "woba_den", "xwoba_num", "swing", "whiff", "rv", "pitches"]].sum()


def outcome_rates(totals: pd.DataFrame) -> pd.DataFrame:
    """Rates per pitcher-season from summed totals."""
    t = totals.groupby(level=[0, 1]).sum().reset_index()
    t["k_pct"] = t["k"] / t["pa"]
    t["bb_pct"] = t["bb"] / t["pa"]
    t["woba"] = t["woba_num"] / t["woba_den"]
    t["xwoba"] = t["xwoba_num"] / t["woba_den"]
    t["whiff_rate"] = t["whiff"] / t["swing"]
    t["rv100"] = t["rv"] / t["pitches"] * 100
    return t[["pitcher", "season", "pa", "k_pct", "bb_pct", "woba", "xwoba", "whiff_rate", "rv100"]]


def season_outcomes(path: str = RAW_FILE) -> pd.DataFrame:
    parts = [chunk_totals(chunk) for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, chunksize=CHUNKSIZE, low_memory=False)]
    return outcome_rates(pd.concat(parts))


def outcome_pairs(ps: pd.DataFrame, outcomes: pd.DataFrame, min_next_pa: int = MIN_NEXT_PA) -> pd.DataFrame:
    """One row per pitcher and pair of consecutive seasons: the earlier season's scores and own results beside the next
    season's results, for pitchers with every score and at least `min_next_pa` batters faced the next season."""
    scored = ps.dropna(subset=["deception_plus", "stuff_plus", "location_plus"]).merge(outcomes, on=["pitcher", "season"], how="left")
    keep = ["pitcher", "season", "deception_plus", "stuff_plus", "location_plus", "qualified"] + [f"{m}_index" for m in MEMBERS] + list(OUTCOMES)
    cur = scored[keep]
    later = outcomes[["pitcher", "season", "pa"] + list(OUTCOMES)].assign(season=lambda d: d["season"] - 1)
    later = later.rename(columns={c: f"next_{c}" for c in ["pa"] + list(OUTCOMES)})
    out = cur.merge(later, on=["pitcher", "season"])
    return out[out["next_pa"] >= min_next_pa].dropna(subset=[f"{m}_index" for m in MEMBERS]).reset_index(drop=True)


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seeds: tuple[int, ...] = CV_SEEDS) -> np.ndarray:
    """Out-of-fold least-squares predictions with folds grouped by pitcher, averaged over random fold assignments."""
    total = np.zeros(len(y))
    for seed in seeds:
        for train, test in GroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed).split(X, y, groups=groups):
            total[test] += LinearRegression().fit(X[train], y[train]).predict(X[test])
    return total / len(seeds)


def r_squared(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2))


def r2_gain_interval(y: np.ndarray, base: np.ndarray, full: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """R2(full) minus R2(base) on the out-of-fold predictions, with a 95% interval from resampling whole pitchers."""
    order = np.argsort(groups, kind="stable")
    y, base, full, groups = y[order], base[order], full[order], groups[order]
    _, starts = np.unique(groups, return_index=True)
    spans = [np.arange(a, b) for a, b in zip(starts, list(starts[1:]) + [len(groups)])]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([spans[i] for i in rng.integers(0, len(spans), len(spans))])
        draws.append(r_squared(y[idx], full[idx]) - r_squared(y[idx], base[idx]))
    return r_squared(y, full) - r_squared(y, base), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def evaluate_outcome(pairs: pd.DataFrame, outcome: str) -> dict:
    """Cross-validated R2 of each predictor set for the next season's `outcome`, and the gain from adding Deception+."""
    d = pairs.dropna(subset=[f"next_{outcome}", outcome]).reset_index(drop=True)
    y, groups = d[f"next_{outcome}"].to_numpy(), d["pitcher"].to_numpy()
    sets = {"stuff_location": ["stuff_plus", "location_plus"], "stuff_location_deception": ["stuff_plus", "location_plus", "deception_plus"],
            "deception": ["deception_plus"], "own": [outcome], "own_stuff_location": [outcome, "stuff_plus", "location_plus"],
            "own_stuff_location_deception": [outcome, "stuff_plus", "location_plus", "deception_plus"]}
    pred = {name: oof_predictions(d[cols].to_numpy(), y, groups) for name, cols in sets.items()}

    def gain(base: str, full: str) -> dict:
        delta, lo, hi = r2_gain_interval(y, pred[base], pred[full], groups)
        return {"gain": delta, "lo": lo, "hi": hi}

    return {"n": int(len(d)), "pitchers": int(d["pitcher"].nunique()), "r2": {k: r_squared(y, p) for k, p in pred.items()},
            "deception_over_stuff_location": gain("stuff_location", "stuff_location_deception"),
            "deception_over_own_stuff_location": gain("own_stuff_location", "own_stuff_location_deception")}


def member_coefficients(pairs: pd.DataFrame, outcome: str) -> dict:
    """Standardized least-squares coefficients (change in the next season's outcome per SD of each predictor), standard
    errors clustered on pitcher."""
    predictors = ["stuff_plus", "location_plus"] + [f"{m}_index" for m in MEMBERS]
    d = pairs.dropna(subset=[f"next_{outcome}"] + predictors)
    X = (d[predictors] - d[predictors].mean()) / d[predictors].std()
    fit = sm.OLS(d[f"next_{outcome}"].to_numpy(), sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": d["pitcher"].to_numpy()})
    ci = fit.conf_int()
    return {"n": int(len(d)), **{p: {"coef": float(fit.params[p]), "p": float(fit.pvalues[p]), "lo": float(ci.loc[p, 0]), "hi": float(ci.loc[p, 1])} for p in predictors}}


def timing_rule(coefficients: dict, share: float = TIMING_RULE_SHARE) -> dict:
    """The demotion rule for timing: on both strikeout rate and xwOBA, is the upper end of its interval under `share` of
    whiff's coefficient, once signs are aligned so that positive is better for the pitcher?"""
    detail = {}
    for outcome in ("k_pct", "xwoba"):
        sign = OUTCOMES[outcome][1]
        c = coefficients[outcome]
        whiff = sign * c["whiff_index"]["coef"]
        timing_hi = max(sign * c["timing_index"]["lo"], sign * c["timing_index"]["hi"])
        detail[outcome] = {"timing_upper": timing_hi, "whiff_coefficient": whiff, "ratio": timing_hi / whiff if whiff > 0 else float("nan"),
                           "below_rule": bool(whiff > 0 and timing_hi < share * whiff)}
    return {"share": share, "outcomes": detail, "demote": bool(all(v["below_rule"] for v in detail.values()))}


def stability_benchmarks(ps: pd.DataFrame, outcomes: pd.DataFrame, min_pa: int = MIN_NEXT_PA) -> dict:
    """Year-over-year correlation of each metric among pitchers qualified in both seasons with at least `min_pa` batters faced."""
    d = ps.merge(outcomes, on=["pitcher", "season"], how="left")
    d = d[d["qualified"] & (d["pa"] >= min_pa)]
    cols = list(STABILITY_METRICS)
    pairs = consecutive_pairs(d[["pitcher", "season"] + cols], cols)
    return {"n": int(len(pairs)), "r": {c: float(pairs[f"{c}_1"].corr(pairs[f"{c}_2"])) for c in cols}}


if __name__ == "__main__":
    ps = pd.read_csv("output/pitcher_season.csv")
    print("computing each pitcher-season's results from the pitch file...", flush=True)
    outcomes = season_outcomes()
    pairs = outcome_pairs(ps, outcomes)
    print(f"{len(pairs)} pairs of consecutive seasons ({pairs['pitcher'].nunique()} pitchers) with at least {MIN_NEXT_PA} batters faced the next season", flush=True)

    result = {"min_next_pa": MIN_NEXT_PA, "outcomes": {}, "coefficients": {}}
    for outcome, (label, _) in OUTCOMES.items():
        e = evaluate_outcome(pairs, outcome)
        result["outcomes"][outcome] = {"label": label, **e}
        g1, g2 = e["deception_over_stuff_location"], e["deception_over_own_stuff_location"]
        print(f"{label:28s} n={e['n']}  Stuff+/Location+ {e['r2']['stuff_location']:.3f} -> +Deception+ {e['r2']['stuff_location_deception']:.3f} "
              f"({g1['gain']:+.3f} [{g1['lo']:+.3f}, {g1['hi']:+.3f}]);  own+Stuff+/Location+ {e['r2']['own_stuff_location']:.3f} -> "
              f"{e['r2']['own_stuff_location_deception']:.3f} ({g2['gain']:+.3f} [{g2['lo']:+.3f}, {g2['hi']:+.3f}])", flush=True)
        result["coefficients"][outcome] = member_coefficients(pairs, outcome)
    result["timing_rule"] = timing_rule(result["coefficients"])
    result["stability"] = stability_benchmarks(ps, outcomes)
    for outcome in ("k_pct", "xwoba"):
        t = result["coefficients"][outcome]["timing_index"]
        w = result["coefficients"][outcome]["whiff_index"]
        print(f"{OUTCOMES[outcome][0]:16s} per SD: timing {t['coef']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}], whiff {w['coef']:+.4f} [{w['lo']:+.4f}, {w['hi']:+.4f}]")
    rule = result["timing_rule"]
    print(f"timing rule (demote if both upper bounds are under {rule['share']:.0%} of whiff's coefficient): "
          f"{'MET, demote' if rule['demote'] else 'not met, keep'}  " + ", ".join(f"{k}: ratio {v['ratio']:+.2f}" for k, v in rule["outcomes"].items()))
    print("year-over-year (qualified pairs): " + ", ".join(f"{STABILITY_METRICS[k]} {v:.2f}" for k, v in result["stability"]["r"].items()))
    with open("output/outcome_validation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Saved outcome_validation.json. Done.")
