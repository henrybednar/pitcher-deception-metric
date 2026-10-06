"""
Pitcher Deception Project: whiff and chase scores by pitch type
================================================================
Deception+ is one number per pitcher-season. This step scores the two biggest members, whiff and chase, within one
pitch type at a time (a pitcher's slider against every other slider), with the same shrinkage, 100/10 scale, game
bootstrap interval and split-half reliability as the season scores (reliability_and_ci.process_outcome), so the
leaderboard can show where a pitcher's score comes from. Per-type samples are small, so the reliability by type is
written next to the scores: it says how much of a per-type score to believe.

Run after fit_called_strike.py. Writes pitch_type_scores.csv (one row per pitcher, season and pitch type, with the
sample, index and 95% interval for whiff and chase) and pitch_type_scores.json (the reliability and sample by pitch type).
"""

import json

import pandas as pd

from reliability_and_ci import BINARY_OUTCOMES, MIN_N_FOR_SCORE, process_outcome

PREDICTIONS_FILE = "output/per_pitch_predictions.csv"
LABELS = ("whiff", "chase")
PITCH_TYPES = ("FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS")
SCORE_COLS = ["{label}_n", "{label}_index", "{label}_ci_lo", "{label}_ci_hi"]
PITCH_TYPE_NAMES = {"FF": "four-seam fastballs", "SI": "sinkers", "FC": "cutters", "SL": "sliders", "ST": "sweepers",
                    "CU": "curveballs", "KC": "knuckle curves", "CH": "changeups", "FS": "splitters"}
SHOW_MIN_N = 50   # the leaderboard shows a per-type index only from this many swings (whiff) or pitches out of the zone (chase)


def score_pitch_type(df: pd.DataFrame, pitch_type: str, label: str) -> tuple[pd.DataFrame, dict]:
    """The label's scores for one pitch type: one row per pitcher-season, and the reliability summary."""
    sub = df[df["pitch_type"] == pitch_type]
    result, rel, _ = process_outcome(sub, label, BINARY_OUTCOMES[label], is_binary=True)
    keep = ["pitcher", "season"] + [c.format(label=label) for c in SCORE_COLS]
    scored = result[keep].dropna(subset=[f"{label}_index"])
    reliability = float(rel["r_full_spearman_brown"])
    summary = {"reliability": reliability if reliability == reliability else None, "n_pitcher_seasons": int(len(scored)),
               "median_n": float(scored[f"{label}_n"].median()), "n_half_reliable": int(rel["n_half_reliable"]),
               "design_effect": float(rel["design_effect"])}
    return scored.assign(pitch_type=pitch_type), summary


def score_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every pitch type's scores in one table, and the summary by pitch type and label."""
    tables, summary = [], {}
    for pitch_type in PITCH_TYPES:
        per_label = {}
        merged = None
        for label in LABELS:
            scored, per_label[label] = score_pitch_type(df, pitch_type, label)
            merged = scored if merged is None else merged.merge(scored, on=["pitcher", "season", "pitch_type"], how="outer")
        tables.append(merged)
        summary[pitch_type] = per_label
    return pd.concat(tables, ignore_index=True), summary


def leaderboard_lists(table: pd.DataFrame) -> dict[tuple[int, int], list[list]]:
    """For each (pitcher, season), the pitch types worth showing as [pitch type, whiff n, whiff index, chase n, chase index],
    most used first. An index is left None below SHOW_MIN_N, and a type with neither index is left out."""
    shown: dict[tuple[int, int], list[list]] = {}
    for row in table.itertuples(index=False):
        entry = [row.pitch_type]
        for label in LABELS:
            n, index = getattr(row, f"{label}_n"), getattr(row, f"{label}_index")
            has_n = pd.notna(n)
            entry += [int(n) if has_n else None, round(float(index), 1) if has_n and n >= SHOW_MIN_N and pd.notna(index) else None]
        if entry[2] is None and entry[4] is None:
            continue
        shown.setdefault((int(row.pitcher), int(row.season)), []).append(entry)
    for lists in shown.values():
        lists.sort(key=lambda e: -((e[1] or 0) + (e[3] or 0)))
    return shown


if __name__ == "__main__":
    print("loading per-pitch predictions...", flush=True)
    predictions = pd.read_csv(PREDICTIONS_FILE)
    table, by_type = score_all(predictions)
    table.to_csv("output/pitch_type_scores.csv", index=False)
    with open("output/pitch_type_scores.json", "w", encoding="utf-8") as f:
        json.dump({"min_n_for_score": MIN_N_FOR_SCORE, "types": by_type}, f, indent=2)
    print(f"\nSaved pitch_type_scores.csv: {len(table):,} pitcher-season-pitch-type rows. Done.")
    for pitch_type, per_label in by_type.items():
        print(f"{pitch_type}: " + ", ".join(f"{label} r={v['reliability'] if v['reliability'] is not None else float('nan'):.2f} "
                                            f"(n={v['n_pitcher_seasons']:,}, median {v['median_n']:.0f})" for label, v in per_label.items()))
