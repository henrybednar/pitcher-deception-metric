"""
Pitcher Deception Project — pitch-level sequencing driver analysis
========================================================================
Every prior driver analysis regressed a SEASON-level score on SEASON-level
proxies (~1,500 rows, e.g. "average velocity gap from previous pitch") and
found near-zero R² for sequencing/tunneling effects. That's a weak test —
season aggregates wash out pitch-to-pitch variation entirely, and 1,500
rows isn't much power to detect small effects.

This instead asks the question at the level it actually happens: for each
of the 6 outcomes, does the ACTUAL preceding pitch in that at-bat (not a
season average) predict the CURRENT pitch's residual (actual − full
expectation)? Millions of rows instead of ~1,500, and real per-at-bat
sequencing features instead of proxies:
  - the pair features in pitch_pairs.py: velocity, plate-location and movement change from the
    previous pitch, whether the pitch type changed, this pair's actual spin-mirror score (not a
    season mean), tunnel separation, and the interactions with a pitch-type change

None of them is in any model's expectation, so a residual that depends on them is a sequencing
effect the metric credits to the pitcher.

This is the more powerful version of the same tunneling/sequencing
hypothesis the project already tested and found weak — if it's still null
here, that's a much stronger negative result than the season-level check.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from build_pitch_table import PITCH_LEVEL_FILE, load_pitch_data, read_aligned_predictions
from pitch_pairs import PAIR_FEATURES
from reliability_and_ci import BINARY_OUTCOMES, CONTINUOUS_OUTCOMES

OUTCOMES = {**BINARY_OUTCOMES, **CONTINUOUS_OUTCOMES}
MAX_ROWS_PER_OUTCOME = 300_000
RANDOM_STATE = 42


if __name__ == "__main__":
    print("loading full pitch-level data...", flush=True)
    df = load_pitch_data(PITCH_LEVEL_FILE)
    preds = read_aligned_predictions(df)

    needed = {"is_take", "is_called_strike"}
    for label, spec in OUTCOMES.items():
        needed |= {spec["target"], f"{label}_expected_full"}
    # every outcome's subset lambda reads is_swing / is_bip / is_whiff style flags that
    # load_pitch_data already builds, so only predictions and targets come from the file
    for c in sorted(needed):
        df[c] = preds[c].values

    results = {}
    for label, spec in OUTCOMES.items():
        mask = spec["subset"](df) & df[PAIR_FEATURES].notna().all(axis=1)
        sub = df.loc[mask, ["pitcher", spec["target"], f"{label}_expected_full"] + PAIR_FEATURES].dropna()
        if len(sub) > MAX_ROWS_PER_OUTCOME:
            sub = sub.sample(MAX_ROWS_PER_OUTCOME, random_state=RANDOM_STATE)
        resid = sub[spec["target"]] - sub[f"{label}_expected_full"]

        # GroupShuffleSplit by pitcher — a random row split would let the
        # same pitcher's pitches (highly self-similar in sequencing terms)
        # appear in both train and test.
        train_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=RANDOM_STATE).split(sub, resid, groups=sub["pitcher"]))
        X_train, X_test = sub[PAIR_FEATURES].iloc[train_idx], sub[PAIR_FEATURES].iloc[test_idx]
        y_train, y_test = resid.iloc[train_idx], resid.iloc[test_idx]
        m = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31, learning_rate=0.05, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
        r2 = r2_score(y_test, m.predict(X_test))
        perm = permutation_importance(m, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)

        feats = sorted(
            [{"feature": f, "importance": round(float(imp), 5)} for f, imp in zip(PAIR_FEATURES, perm.importances_mean)],
            key=lambda r: -r["importance"])
        results[label] = {"n": len(sub), "r2": round(float(r2), 4), "features": feats}
        print(f"{label}: n={len(sub):,}, R2={r2:.4f}", flush=True)
        for f in feats:
            print(f"  {f['feature']:22s} importance={f['importance']:.5f}", flush=True)

    import json
    with open("sequencing_driver_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved sequencing_driver_report.json. Done.", flush=True)
