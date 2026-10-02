"""
Pitcher Deception Project: how good are the pitch models?
=========================================================
Deception+ is actual minus expected, so the expectation models are the product and the score is
what they leave over. This step validates them from the out-of-fold predictions the fit_* steps
already wrote, one outcome at a time and one pitch-type model at a time.

  discrimination  AUC for the yes/no outcomes, R-squared for the continuous ones
  log loss        for the yes/no outcomes, against a constant guess of that pitch type's base rate
  calibration     predicted against actual by predicted decile, and the recalibration slope
                  (1.0 is calibrated, above 1 means predictions are too flat)

Every prediction was made with the pitcher's own pitches held out, so none of this is in-sample.

Reads output/per_pitch_predictions.csv. Writes output/model_validation.json, which
build_artifact.py charts and export_site_stats.py quotes.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, r2_score, roc_auc_score

from reliability_and_ci import ALL_OUTCOMES, BINARY_OUTCOMES, COMPOSITE_OUTCOMES, CONTINUOUS_OUTCOMES

PREDICTIONS_FILE = "output/per_pitch_predictions.csv"
OUT_FILE = "output/model_validation.json"
SPECS = {**BINARY_OUTCOMES, **CONTINUOUS_OUTCOMES}
MIN_ROWS_PER_PITCH_TYPE = 1000
N_BINS = 10
P_CLIP = 1e-6
# label, what the model predicts, and the pitches it is scored on
DISPLAY = {
    "whiff": ("Whiff", "whether a swing misses", "swings"),
    "chase": ("Chase", "whether the batter swings", "pitches outside the zone"),
    "weak": ("Weak contact", "whether contact is under 85 mph", "balls in play"),
    "timing": ("Timing", "how far contact is from the hitter's ideal depth (inches)", "contact swings"),
    "whiffmiss": ("Whiff miss distance", "how far the bat misses the ball (log scale)", "swings and misses"),
    "gb": ("Ground ball", "whether a ball in play is a ground ball", "balls in play"),
    "calledstrike": ("Called strike", "whether a taken pitch is called a strike", "pitches the batter takes"),
    "align": ("Horizontal alignment", "how far contact is from the hitter's ideal horizontal reach (inches)", "contact swings"),
}


def constant_log_loss(y: np.ndarray) -> float:
    """Log loss of always guessing the base rate."""
    rate = float(np.clip(np.mean(y), P_CLIP, 1 - P_CLIP))
    return float(-(rate * np.log(rate) + (1 - rate) * np.log(1 - rate)))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p)),
            "baseline_logloss": constant_log_loss(y), "rate": float(np.mean(y))}


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"r2": float(r2_score(y, p)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "baseline_rmse": float(np.std(y))}


def recalibration_slope(y: np.ndarray, p: np.ndarray, binary: bool) -> float:
    """Slope of actual on predicted: a logistic fit on the log odds for yes/no outcomes, a line otherwise."""
    if binary:
        p = np.clip(p, P_CLIP, 1 - P_CLIP)
        fit = LogisticRegression(C=np.inf, max_iter=300).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return float(fit.coef_[0, 0])
    return float(np.polyfit(p, y, 1)[0])


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Mean predicted and mean actual in each predicted-value bin, lowest to highest."""
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    return [{"pred": float(g["p"].mean()), "actual": float(g["y"].mean()), "n": int(len(g))}
            for _, g in frame.groupby("bin")]


def validate_outcome(df: pd.DataFrame, label: str) -> dict:
    spec, expected = SPECS[label], f"{label}_expected_full"
    binary = label in BINARY_OUTCOMES
    scored = spec["subset"](df) & df[expected].notna()
    rows = df.loc[scored, ["pitch_type", spec["target"], expected]].rename(columns={spec["target"]: "y", expected: "p"})
    rows = rows.assign(y=rows["y"].astype(float), p=rows["p"].astype(float))
    metric = binary_metrics if binary else regression_metrics

    by_type = []
    for pitch_type, g in rows.groupby("pitch_type"):
        if len(g) >= MIN_ROWS_PER_PITCH_TYPE:
            by_type.append({"pitch_type": str(pitch_type), "n": int(len(g)),
                            **{k: round(v, 5) for k, v in metric(g["y"].to_numpy(), g["p"].to_numpy()).items()}})
    by_type.sort(key=lambda r: -r["n"])

    y, p = rows["y"].to_numpy(), rows["p"].to_numpy()
    weights = np.array([r["n"] for r in by_type], float)
    if binary:
        overall = {"auc": float(roc_auc_score(y, np.clip(p, P_CLIP, 1 - P_CLIP))),
                   "logloss": float(np.average([r["logloss"] for r in by_type], weights=weights)),
                   "baseline_logloss": float(np.average([r["baseline_logloss"] for r in by_type], weights=weights))}
    else:
        within = y - rows.groupby("pitch_type")["y"].transform("mean").to_numpy()
        overall = {"r2": float(1 - np.sum((y - p) ** 2) / np.sum(within ** 2))}
    return {
        "label": DISPLAY[label][0], "predicts": DISPLAY[label][1], "on": DISPLAY[label][2],
        "kind": "classify" if binary else "regress",
        "in_score": label in COMPOSITE_OUTCOMES, "n": int(len(rows)),
        "overall": {k: round(v, 5) for k, v in overall.items()}, "by_pitch_type": by_type,
        "calibration": {"slope": round(recalibration_slope(y, p, binary), 4),
                        "mean_gap": round(float(np.mean(y) - np.mean(p)), 5),
                        "bins": [{k: round(v, 5) if isinstance(v, float) else v for k, v in b.items()}
                                 for b in calibration_bins(y, p)]},
    }


if __name__ == "__main__":
    print("loading per-pitch predictions...", flush=True)
    df = pd.read_csv(PREDICTIONS_FILE)
    result = {}
    for label in ALL_OUTCOMES:
        result[label] = validate_outcome(df, label)
        r = result[label]
        score = f"AUC {r['overall']['auc']:.3f}, log loss {r['overall']['logloss']:.3f} vs {r['overall']['baseline_logloss']:.3f}" \
            if r["kind"] == "classify" else f"R2 {r['overall']['r2']:.3f}"
        print(f"{label:13s} n={r['n']:>9,}  {score}  calibration slope {r['calibration']['slope']:.3f}", flush=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"outcomes": result}, f, indent=2)
    print(f"Saved {OUT_FILE}. Done.", flush=True)
