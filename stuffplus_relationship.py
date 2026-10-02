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

Reads output/per_pitch_predictions.csv and output/pitcher_pitchtype_season.csv.
Writes output/stuffplus_relationship.json, which build_artifact.py charts.
"""

import json

import numpy as np
import pandas as pd

from reliability_and_ci import BINARY_OUTCOMES, COMPOSITE_OUTCOMES, CONTINUOUS_OUTCOMES

PREDICTIONS_FILE = "output/per_pitch_predictions.csv"
PITCH_TYPE_FILE = "output/pitcher_pitchtype_season.csv"
OUT_FILE = "output/stuffplus_relationship.json"
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


def residual_cells(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per pitcher x season x pitch type: n scored pitches and actual minus expected, on the outcome's own
    scored pitches, with samples too small to be stable dropped."""
    spec, expected = SPECS[label], f"{label}_expected_full"
    scored = spec["subset"](df) & df[expected].notna()
    cells = (df.loc[scored, ["pitcher", "season", "pitch_type", spec["target"], expected]]
             .groupby(["pitcher", "season", "pitch_type"])
             .agg(n=(spec["target"], "size"), actual=(spec["target"], "mean"), expected=(expected, "mean"))
             .reset_index())
    scale = 100.0 if label in BINARY_OUTCOMES else 1.0
    cells["resid"] = (cells["actual"] - cells["expected"]) * scale
    return cells[cells["n"] >= MIN_CELL_N.get(label, DEFAULT_MIN_CELL_N)]


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
    result = relationship(pd.read_csv(PREDICTIONS_FILE),
                          stuff_by_pitch_type(pd.read_csv(PITCH_TYPE_FILE, usecols=[
                              "pitcher", "season", "pitch_type", "stuff_plus_by_pitch_type"])))
    for label, groups in result.items():
        for name in ("fastball", "nonfastball"):
            rows = groups[name]
            print(f"{label:10s} {name:12s} lowest fifth {rows[0]['resid']:+.3f} -> highest fifth {rows[-1]['resid']:+.3f} {groups['unit']}", flush=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"fastball_types": sorted(FASTBALLS), "outcomes": result}, f, indent=2)
    print(f"Saved {OUT_FILE}. Done.", flush=True)
