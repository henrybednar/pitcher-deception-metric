"""
Pitcher Deception Project: next-season projection
=================================================
Projects a pitcher's Deception+ for the next season from the last two. A pitcher-season's score is its own
season's result, shrunk toward 100 for its sample, but it is still partly noise and partly talent that drifts, so
the best guess for next season sits closer to 100 and leans on the season before when there is one:

    one season  (no earlier season scored):   next = a + b x this
    two seasons (this and the one before):    next = a + b1 x this + b2 x previous

The coefficients are fit by least squares on every pair (or triple) of consecutive scored seasons the data holds.
Every projection is out of sample: pitchers are split into folds (five random splits, averaged), and a pitcher's
projection comes from coefficients fit on the other folds' pitchers only, so the backtest (a 2024 or 2025
projection against what happened the next season) is honest, and the projection for the latest season, which has
no outcome yet, is made the same way. The 80% range is the projection plus or minus 1.28 times the standard
deviation of that basis's out-of-sample errors.

Run after reliability_and_ci.py. Writes output/projection.csv (one row per scored pitcher-season, the projection is
for the following season) and output/projection.json (coefficients and backtest), which export_site_stats.py and
export_leaderboard_data.py read.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

CV_SPLITS = 5
CV_SEEDS = (0, 1, 2, 3, 4)
RANGE_Z = 1.2816           # 80% of a normal error distribution
N_BOOT = 1000


def season_rows(ps: pd.DataFrame) -> pd.DataFrame:
    """One row per scored pitcher-season: its score (`cur`), the previous season's (`prev`, NaN if there is none) and
    the next season's (`nxt`, NaN for the latest season), with the pitcher id, season and whether it qualified."""
    scored = ps.dropna(subset=["deception_plus"])[["pitcher", "season", "deception_plus", "qualified"]]
    rows = scored.rename(columns={"deception_plus": "cur"})
    previous = scored[["pitcher", "season", "deception_plus"]].assign(season=lambda d: d["season"] + 1).rename(columns={"deception_plus": "prev"})
    following = scored[["pitcher", "season", "deception_plus"]].assign(season=lambda d: d["season"] - 1).rename(columns={"deception_plus": "nxt"})
    return rows.merge(previous, on=["pitcher", "season"], how="left").merge(following, on=["pitcher", "season"], how="left").reset_index(drop=True)


def fit(rows: pd.DataFrame, features: list[str]) -> LinearRegression:
    train = rows.dropna(subset=features + ["nxt"])
    return LinearRegression().fit(train[features], train["nxt"])


def cross_fitted_projections(rows: pd.DataFrame, seeds: tuple[int, ...] = CV_SEEDS, n_splits: int = CV_SPLITS) -> pd.DataFrame:
    """`proj1` (this season only) for every row and `proj2` (this and the previous season) for rows that have a previous
    season, each from coefficients fit on other pitchers only and averaged over the seeds' fold assignments."""
    sums = {"proj1": np.zeros(len(rows)), "proj2": np.zeros(len(rows))}
    for seed in seeds:
        for train_idx, test_idx in GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(rows, groups=rows["pitcher"]):
            train, test = rows.iloc[train_idx], rows.iloc[test_idx]
            sums["proj1"][test_idx] += fit(train, ["cur"]).predict(test[["cur"]])
            with_prev = test["prev"].notna().to_numpy()
            if with_prev.any():
                sums["proj2"][test_idx[with_prev]] += fit(train, ["cur", "prev"]).predict(test.loc[with_prev, ["cur", "prev"]])
    out = rows.copy()
    out["proj1"] = sums["proj1"] / len(seeds)
    out["proj2"] = np.where(rows["prev"].notna(), sums["proj2"] / len(seeds), np.nan)
    return out


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def add_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """The projection to show: two seasons where the pitcher has them, otherwise one, with its basis and 80% range."""
    out = frame.copy()
    out["proj_basis"] = np.where(out["proj2"].notna(), 2, 1)
    out["projection"] = out["proj2"].where(out["proj2"].notna(), out["proj1"])
    has_next = out["nxt"].notna()
    spread = {basis: float((out.loc[has_next & (out["proj_basis"] == basis), "nxt"] - out.loc[has_next & (out["proj_basis"] == basis), "projection"]).std())
              for basis in (1, 2)}
    half = out["proj_basis"].map(spread) * RANGE_Z
    out["proj_lo"], out["proj_hi"] = out["projection"] - half, out["projection"] + half
    return out


def backtest(frame: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Out-of-sample errors of the projection, of the raw score used as the guess, and of the league average, over every
    row that has a next season; and the one-season against two-season comparison on the rows that have both."""
    pairs = frame.dropna(subset=["nxt"])
    triples = pairs.dropna(subset=["prev"])
    qualified = pairs[pairs["qualified"]]
    # one-season against two-season on the same pitchers, with an interval from resampling pitchers
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(triples), (n_boot, len(triples)))
    boot = [np.sqrt(np.mean((triples["nxt"].to_numpy()[i] - triples["proj1"].to_numpy()[i]) ** 2))
            - np.sqrt(np.mean((triples["nxt"].to_numpy()[i] - triples["proj2"].to_numpy()[i]) ** 2)) for i in idx]
    return {
        "pairs": int(len(pairs)), "triples": int(len(triples)), "qualified_pairs": int(len(qualified)),
        "rmse_projection": rmse(pairs["nxt"], pairs["projection"]),
        "rmse_raw_score": rmse(pairs["nxt"], pairs["cur"]),
        "rmse_league_average": rmse(pairs["nxt"], pd.Series(100.0, index=pairs.index)),
        "r_projection": float(pairs["nxt"].corr(pairs["projection"])),
        "r_raw_score": float(pairs["nxt"].corr(pairs["cur"])),
        "qualified_rmse_projection": rmse(qualified["nxt"], qualified["projection"]),
        "qualified_rmse_raw_score": rmse(qualified["nxt"], qualified["cur"]),
        "triples_rmse_one_season": rmse(triples["nxt"], triples["proj1"]),
        "triples_rmse_two_seasons": rmse(triples["nxt"], triples["proj2"]),
        "two_season_gain": float(np.mean(boot)), "two_season_gain_lo": float(np.percentile(boot, 2.5)),
        "two_season_gain_hi": float(np.percentile(boot, 97.5)),
        "coverage_80": float(((pairs["nxt"] >= pairs["proj_lo"]) & (pairs["nxt"] <= pairs["proj_hi"])).mean()),
    }


def coefficients(frame: pd.DataFrame) -> dict:
    """The least-squares coefficients on all the data, for reading (the projections themselves are cross-fitted)."""
    one, two = fit(frame, ["cur"]), fit(frame, ["cur", "prev"])
    return {"one_season": {"intercept": float(one.intercept_), "this": float(one.coef_[0])},
            "two_seasons": {"intercept": float(two.intercept_), "this": float(two.coef_[0]), "previous": float(two.coef_[1])}}


if __name__ == "__main__":
    ps = pd.read_csv("output/pitcher_season.csv")
    frame = add_projection(cross_fitted_projections(season_rows(ps)))
    result = {"backtest": backtest(frame), "coefficients": coefficients(frame),
              "latest_season": int(frame["season"].max()), "rows": int(len(frame))}
    b, c = result["backtest"], result["coefficients"]
    print(f"{b['pairs']} projections with an outcome ({b['triples']} on two seasons). RMSE: projection {b['rmse_projection']:.2f}, "
          f"raw score {b['rmse_raw_score']:.2f}, league average {b['rmse_league_average']:.2f}")
    print(f"two seasons against one on the same {b['triples']} pitchers: {b['triples_rmse_one_season']:.3f} to {b['triples_rmse_two_seasons']:.3f} "
          f"(RMSE gain {b['two_season_gain']:+.3f} [{b['two_season_gain_lo']:+.3f}, {b['two_season_gain_hi']:+.3f}]); 80% range covers {b['coverage_80']:.0%}")
    print(f"one season: {c['one_season']['intercept']:.1f} + {c['one_season']['this']:.2f} x this; "
          f"two seasons: {c['two_seasons']['intercept']:.1f} + {c['two_seasons']['this']:.2f} x this + {c['two_seasons']['previous']:.2f} x previous")
    frame[["pitcher", "season", "projection", "proj_lo", "proj_hi", "proj_basis"]].to_csv("output/projection.csv", index=False)
    with open("output/projection.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Saved projection.csv and projection.json. Done.")
