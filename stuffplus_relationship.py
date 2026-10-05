"""
Pitcher Deception Project: does Deception+ track Stuff+?
=========================================================
Deception+ correlates about 0.4 with FanGraphs Stuff+ although every model controls for the
pitch's own velocity, spin, movement, release point and location. This step shows where that
relationship sits. For each composite member it takes pitcher x season x pitch-type samples,
ranks them into fifths by that pitch type's own Stuff+ (so a slider is ranked against other
sliders), and reports the average actual-minus-expected result in each fifth. Fastballs
(four-seamers and sinkers) are kept apart from every other pitch type.

Stuff+ is never a model input. It is read here only to describe the score, the same way the
dashboard's scatter reads it. FanGraphs' export has no sweeper Stuff+, so sweepers are left out.

Two more views answer the question of whether sequencing creates the link (pitch order, how one pitch
sets up the next), as opposed to Stuff+ simply tracking the results of the same pitch:
  by pitch type   the slope of the residual on that pitch type's own Stuff+, for each pitch type on its
                  own (every pitcher's four-seamer against every other pitcher's four-seamer, and so on)
                  and pooled within type. A pitch type's residual never mixes in another pitch type.
  first pitch     the same slopes using only the first pitch of each plate appearance. There is no previous
                  pitch to set it up, so nothing about sequencing can reach those results.
The slope is the change in actual-minus-expected result per 10 Stuff+ points, weighted by sample size,
with standard errors clustered on pitcher.

Reads output/per_pitch_predictions.csv, output/pitcher_pitchtype_season.csv and the raw pitch file (for
pitch_number). Writes output/stuffplus_relationship.json, which build_artifact.py charts.
"""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from raw_alignment import regular_season_rows
from reliability_and_ci import BINARY_OUTCOMES, COMPOSITE_OUTCOMES, CONTINUOUS_OUTCOMES

PREDICTIONS_FILE = "output/per_pitch_predictions.csv"
PITCH_TYPE_FILE = "output/pitcher_pitchtype_season.csv"
RAW_FILE = "raw/statcast_pitch_level_2024_2026.csv"
OUT_FILE = "output/stuffplus_relationship.json"
MIN_CELLS_PER_TYPE = 30
FIRST_PITCH_MIN_N_SCALE = 0.5            # a first-pitch sample is about a quarter the size, so the minimum is halved
SPECS = {**BINARY_OUTCOMES, **CONTINUOUS_OUTCOMES}
FASTBALLS = frozenset({"FF", "SI"})
N_FIFTHS = 5
DEFAULT_MIN_CELL_N = 30
MIN_CELL_N = {"whiff": 50, "chase": 50, "timing": 40}
CONTINUOUS_UNITS = {"timing": "in", "align": "in", "whiffmiss": "log"}


def outcome_unit(label: str) -> str:
    """Binary outcomes are shown in percentage points, continuous ones in their own units."""
    return "pp" if label in BINARY_OUTCOMES else CONTINUOUS_UNITS.get(label, "")


def stuff_by_pitch_type(table: pd.DataFrame) -> pd.DataFrame:
    """pitcher x season x pitch type Stuff+, with FanGraphs' forkball and slurve folded into the
    split-finger and sweeper types the models use."""
    known = table.dropna(subset=["stuff_plus_by_pitch_type"])
    known = known.assign(pitch_type=known["pitch_type"].replace({"FO": "FS", "SV": "ST"}))
    return (known.groupby(["pitcher", "season", "pitch_type"], as_index=False)["stuff_plus_by_pitch_type"].mean()
            .rename(columns={"stuff_plus_by_pitch_type": "stuff"}))


def residual_cells(df: pd.DataFrame, label: str, restrict: pd.Series | None = None,
                   min_n_scale: float = 1.0) -> pd.DataFrame:
    """Per pitcher x season x pitch type: n scored pitches and actual minus expected, on the outcome's own
    scored pitches (optionally only those where `restrict` is true), with samples too small to be stable
    dropped. `min_n_scale` shrinks the minimum sample for a restricted subset."""
    spec, expected = SPECS[label], f"{label}_expected_full"
    scored = spec["subset"](df) & df[expected].notna()
    if restrict is not None:
        scored = scored & restrict
    cells = (df.loc[scored, ["pitcher", "season", "pitch_type", spec["target"], expected]]
             .groupby(["pitcher", "season", "pitch_type"])
             .agg(n=(spec["target"], "size"), actual=(spec["target"], "mean"), expected=(expected, "mean"))
             .reset_index())
    scale = 100.0 if label in BINARY_OUTCOMES else 1.0
    cells["resid"] = (cells["actual"] - cells["expected"]) * scale
    return cells[cells["n"] >= max(15, round(MIN_CELL_N.get(label, DEFAULT_MIN_CELL_N) * min_n_scale))]


def cluster_weighted_mean(values, weights, clusters) -> tuple[float, float]:
    """Weighted mean and its cluster-robust standard error. Clusters are pitchers, who appear in both
    seasons and in several pitch types, so their samples are not independent draws."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    mean = float((w * v).sum() / w.sum())
    score_by_cluster = pd.Series(w * (v - mean)).groupby(np.asarray(clusters)).sum()
    return mean, float(np.sqrt((score_by_cluster ** 2).sum()) / w.sum())


def fifths_within_pitch_type(cells: pd.DataFrame) -> pd.Series:
    """1 (lowest Stuff+) to 5 (highest) within each pitch type."""
    rank = cells.groupby("pitch_type")["stuff"].rank(method="first")
    size = cells.groupby("pitch_type")["stuff"].transform("size")
    return (((rank - 1) * N_FIFTHS // size) + 1).astype(int)


def fifths_table(cells: pd.DataFrame) -> list[dict]:
    rows = []
    for fifth, group in cells.groupby(fifths_within_pitch_type(cells)):
        mean, se = cluster_weighted_mean(group["resid"], group["n"], group["pitcher"])
        rows.append({"fifth": int(fifth), "cells": int(len(group)), "mean_stuff": round(float(group["stuff"].mean()), 1),
                     "resid": round(mean, 3), "se": round(se, 3)})
    return rows


def stuff_slope(cells: pd.DataFrame, x: str = "stuff", y: str = "resid") -> dict:
    """Weighted least-squares slope of the residual on Stuff+, per 10 Stuff+ points, weighted by sample
    size, with a 95% interval from standard errors clustered on pitcher, and the weighted correlation."""
    weights = cells["n"].to_numpy(float)
    clusters = pd.factorize(cells["pitcher"])[0]
    fit = sm.WLS(cells[y].to_numpy(float), sm.add_constant(cells[x].to_numpy(float)), weights=weights).fit(
        cov_type="cluster", cov_kwds={"groups": clusters})
    slope, se = float(fit.params[1]), float(fit.bse[1])
    xm, ym = np.average(cells[x], weights=weights), np.average(cells[y], weights=weights)
    cov = np.average((cells[x] - xm) * (cells[y] - ym), weights=weights)
    r = cov / np.sqrt(np.average((cells[x] - xm) ** 2, weights=weights) * np.average((cells[y] - ym) ** 2, weights=weights))
    return {"per_10": 10 * slope, "se": 10 * se, "lo": 10 * (slope - 1.96 * se), "hi": 10 * (slope + 1.96 * se),
            "r": float(r), "cells": int(len(cells)), "pitchers": int(cells["pitcher"].nunique()),
            "stuff_sd": float(cells[x].std())}


def slopes_by_pitch_type(cells: pd.DataFrame) -> dict[str, dict]:
    """stuff_slope for each pitch type with enough samples, plus "all": one slope pooled across types after
    taking each type's own weighted means out of Stuff+ and the residual, so a type that sits lower on both
    (changeups) cannot masquerade as a relationship."""
    result = {pitch_type: stuff_slope(group) for pitch_type, group in cells.groupby("pitch_type")
              if len(group) >= MIN_CELLS_PER_TYPE}
    kept = cells[cells["pitch_type"].isin(result)]
    if kept.empty:
        return result
    centred = kept.copy()
    for column in ("stuff", "resid"):
        means = kept.groupby("pitch_type").apply(lambda g: np.average(g[column], weights=g["n"]), include_groups=False)
        centred[column] = kept[column] - kept["pitch_type"].map(means)
    result["all"] = stuff_slope(centred)
    return result


def first_pitch_flags(raw_path: str, scored: pd.DataFrame) -> pd.Series:
    """True for the first pitch of each plate appearance, aligned by position to the scored pitches (see
    raw_alignment.regular_season_rows for the row-by-row check)."""
    return regular_season_rows(raw_path, ["pitch_number"], scored)["pitch_number"] == 1


def pitch_type_views(df: pd.DataFrame, stuff: pd.DataFrame, first_pitch: pd.Series) -> dict:
    """For each composite member: slopes by pitch type over all pitches, and over first pitches only."""
    out = {}
    for label in COMPOSITE_OUTCOMES:
        every = residual_cells(df, label).merge(stuff, on=["pitcher", "season", "pitch_type"], how="inner")
        first = residual_cells(df, label, restrict=first_pitch, min_n_scale=FIRST_PITCH_MIN_N_SCALE).merge(
            stuff, on=["pitcher", "season", "pitch_type"], how="inner")
        out[label] = {"unit": outcome_unit(label), "all_pitches": slopes_by_pitch_type(every),
                      "first_pitch": slopes_by_pitch_type(first)}
    return out


def relationship(df: pd.DataFrame, stuff: pd.DataFrame) -> dict:
    out = {}
    for label in COMPOSITE_OUTCOMES:
        cells = residual_cells(df, label).merge(stuff, on=["pitcher", "season", "pitch_type"], how="inner")
        is_fastball = cells["pitch_type"].isin(FASTBALLS)
        out[label] = {"unit": outcome_unit(label), "fastball": fifths_table(cells[is_fastball]),
                      "nonfastball": fifths_table(cells[~is_fastball])}
    return out


if __name__ == "__main__":
    print("loading per-pitch predictions...", flush=True)
    predictions = pd.read_csv(PREDICTIONS_FILE)
    stuff_table = stuff_by_pitch_type(pd.read_csv(PITCH_TYPE_FILE, usecols=[
        "pitcher", "season", "pitch_type", "stuff_plus_by_pitch_type"]))
    result = relationship(predictions, stuff_table)
    by_type = pitch_type_views(predictions, stuff_table, first_pitch_flags(RAW_FILE, predictions))
    for label, views in by_type.items():
        print(f"\n{label}: change in result per +10 Stuff+ ({views['unit']}), all pitches / first pitch only", flush=True)
        for pitch_type in sorted(views["all_pitches"], key=lambda t: (t == "all", t)):
            a, f = views["all_pitches"][pitch_type], views["first_pitch"].get(pitch_type)
            first_txt = f"{f['per_10']:+.2f} [{f['lo']:+.2f}, {f['hi']:+.2f}] n={f['cells']}" if f else "too few"
            print(f"  {pitch_type:3s} {a['per_10']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}] n={a['cells']}   |  {first_txt}", flush=True)
    for label, groups in result.items():
        for name in ("fastball", "nonfastball"):
            rows = groups[name]
            print(f"{label:10s} {name:12s} lowest fifth {rows[0]['resid']:+.3f} -> highest fifth {rows[-1]['resid']:+.3f} {groups['unit']}", flush=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"fastball_types": sorted(FASTBALLS), "outcomes": result, "by_pitch_type": by_type}, f, indent=2)
    print(f"Saved {OUT_FILE}. Done.", flush=True)
