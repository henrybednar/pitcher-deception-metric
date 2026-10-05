"""
Pitcher Deception Project: the "full" model tier
================================================
Every scored outcome (whiff, chase, ground ball, weak contact, and the bat-tracking and called-strike
outcomes in the fit_* steps after this one) is scored against a "full" expectation: stuff + location
+ opponent + context, so confounds that vary by opponent or context do not leak into the pitcher's
residual.
  - the batter's same-season tendency for that exact outcome, plus "stand" (handedness):
    opponent quality and platoon.
  - the catcher's same-season tendency for that exact outcome. Framing is the obvious catcher
    confound, but game-calling could plausibly leak into whiff and chase too, so there is no
    principled reason to control for the catcher on one outcome and not the others.
  - balls and strikes (count state), home_team (park identity) and same_hand (pitcher and
    batter on the same side).

The Deception+ residual is actual minus the "full" expectation. reliability_and_ci.py's
game_level_table() defaults to the "full" suffix.

GROUPKFOLD, not plain KFold: 85%+ of four-seam fastballs come from pitchers who threw in BOTH
2025 and 2026. A plain shuffled KFold lets a pitcher's own pitches land in both train and test,
so the model can learn "pitches shaped exactly like Pitcher X's get extra whiffs" from that
pitcher's other-season pitches, which suppresses the measured residual in proportion to how
distinctive and stable the pitcher's stuff is. GroupKFold guarantees no pitcher ever splits.

The batter and catcher tendencies are built INSIDE each fold from training pitchers only (see
tendencies.py for the two leaks that closes).

Squared-error loss is the default for regression targets: the timing-miss target's skew is 0.24
(near-symmetric) and it is physically bounded (about 72 in), so no heavy tail. Re-check the skew
before reusing fit_tier on a heavy-tailed target.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from build_pitch_table import PITCH_LEVEL_FILE, load_pitch_data, read_aligned_predictions, save_predictions
from physics_features import PHYSICS_FEATURES
from tendencies import build_totals, tendency_for_rows

MIN_N_FOR_MODEL = 5000
# A pitcher's expectation depends on which other pitchers share their cross-validation fold, and with one
# fixed assignment that dependence shows up as noise in the score: two random assignments gave component
# indexes that differed by an SD of 2.5 to 3.0 index points (corr 0.96 to 0.97, largest gaps 12 to 32), so
# each run carries about 2 points of its own. The scored outcomes average the out-of-fold predictions of
# several random grouped splits, which cuts that by about 1/sqrt(n). Every prediction stays out of fold.
# Whiff, two replicates each of different seeds: the SD of the difference between replicates in the whiff
# index was 1.46 points with three seeds, 1.06 with six, and 1.28 with three seeds on ten folds (which
# takes as long as six seeds on five). Out-of-fold log loss, 0.4212 / 0.4205 / 0.4211. Year-over-year of
# the whiff index did not move (0.51 to 0.52 in all three), since a pitcher's two seasons share a fold.
FOLD_SEEDS = (0, 1, 2, 3, 4, 5)
STUFF_FEATURES = [
    "release_speed", "release_spin_rate", "ivb_in", "arm_side_break_in",
    "vaa", "haa", "release_extension", "effective_speed", "arm_angle",
    "release_pos_z", "release_pos_x_armside", "p_throws",
] + PHYSICS_FEATURES
LOCATION_FEATURES = ["plate_x_armside", "plate_z_norm"]
OUTCOMES = {
    "whiff": dict(subset="is_swing", target="is_whiff", kind="classify"),
    "chase": dict(subset="not_in_zone", target="is_swing", kind="classify"),
    "gb": dict(subset="is_bip", target="is_gb", kind="classify"),
    "weak": dict(subset="is_bip", target="is_weak", kind="classify"),
}
CONTEXT_FEATURES = ["stand", "balls", "strikes", "home_team", "same_hand", "season", "pitch_count_in_appearance",
                    "times_faced_this_game", "outs", "runners_on", "score_diff"]
# season is a fixed effect, not a talent signal: the 2025-26 pull carries a real leaguewide
# calibration gap between the two years (e.g. whiff actual-minus-expected +1.2 points in 2025,
# -0.8 in 2026, before this feature existed), and without a way to see which season a pitch is
# from, the model can't help but read that gap as pitcher skill. Adding it lets the model absorb
# the level shift instead.
# pitch_count_in_appearance (build_pitch_table.py) is the same idea for role and in-game fatigue:
# without it the model has no way to know a pitch was the reliever's 8th of the night or the
# starter's 88th, so it read most of that gap as pitcher skill too. A held-out test found this
# closed the reliever-vs-starter whiff residual gap by about 94% (~1.3pp actual-minus-expected to
# ~0.1pp), for a real if small held-out AUC gain (+0.0004, 95% interval +0.0000 to +0.0007).
CATEGORICAL = ["p_throws", "stand", "home_team", "pitch_type", "season"]
# A pitch type with too few rows for its own model shares a similar type's model, with pitch_type as a
# feature. Knuckle curves alone (about 3.5k rows for ground ball, weak contact and whiff miss) had no
# skill, log loss 0.694 against 0.693 for a constant. With curveballs they gain 0.025, 0.034 and 0.063
# (log loss, log loss, squared error), and curveballs gain a little too.
POOLED_WITH = {"KC": "CU"}
# name -> (key_col, split_by_hand). batter_tendency_same_hand matches each row against only the
# training batters' outcomes vs pitchers who share ITS OWN pitcher's handedness, instead of one
# blended number across both. A held-out test found this adds real signal for whiff (+0.0008 AUC,
# 95% interval +0.0003 to +0.0013) and is neutral for chase (interval includes zero); kept as an
# addition alongside the blended batter_tendency, not a replacement, since catchers have no
# platoon-split analog and the blended number still carries real signal on its own.
# batter_pitch_type_tendency is the batter's rate for that outcome against the same pitch type, which the
# pooled tendency averages away (a hitter who chases sliders is not one who chases fastballs). Held out
# by pitcher on whiff and chase, one seed: log loss 0.4252 to 0.4247 for whiff (gain +0.00053, 95%
# interval +0.00026 to +0.00078) and 0.4159 to 0.4139 for chase (+0.0020, +0.0018 to +0.0023).
# batter_count_tendency (batter by strikes) and batter_zone_tendency (batter by in or out of the zone) were
# added the same way, each on top of the one before it: chase +0.0015 (+0.0012 to +0.0018) from the count,
# whiff +0.0010 (+0.0007 to +0.0012) from the zone. The reverse pairings gained nothing (whiff from the
# count +0.0002, interval including zero; chase from the zone is the batter tendency again, since every chase
# pitch is out of the zone), and the catcher by pitch type lost 0.0001 on chase.
TENDENCY_KEYS = {
    "batter_tendency": ("batter", False),
    "catcher_tendency": ("fielder_2", False),
    "batter_tendency_same_hand": ("batter", True),
    "batter_pitch_type_tendency": ("batter_pitch_type", False),
    "batter_count_tendency": ("batter_strikes", False),
    "batter_zone_tendency": ("batter_zone", False),
}


def add_same_hand(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit platoon-matchup feature (p_throws == stand). p_throws and stand were already both
    features, so the model could in principle learn the platoon interaction, but it was not
    reliably finding it: same_hand alone carried more permutation importance than p_throws and
    stand combined, and adding it explicitly raised AUC (0.748 to 0.750 on four-seamers). Without
    it, genuine platoon effects were partly leaking into the residual as pitcher-specific deception."""
    df["same_hand"] = (df["p_throws"].astype(str) == df["stand"].astype(str)).astype(int)
    return df


def prepare_context(df: pd.DataFrame) -> pd.DataFrame:
    """Columns the full tier reads beyond load_pitch_data's."""
    df["not_in_zone"] = ~df["is_in_zone"]
    df["stand"] = df["stand"].astype("category")
    df["home_team"] = df["home_team"].astype("category")
    batter = df["batter"].astype(str)
    df["batter_pitch_type"] = batter + "_" + df["pitch_type"].astype(str)
    df["batter_strikes"] = batter + "_" + df["strikes"].astype(str)
    df["batter_zone"] = batter + "_" + df["is_in_zone"].astype(str)
    return add_same_hand(df)


def make_model(kind: str, feature_cols: list[str]):
    # l2_regularization=3.0: a hyperparameter sweep (four-seam whiff, held-out pitchers) found this
    # beats the old 1.0 by +2.66e-4 logloss with a 95% CI entirely above zero ([+0.36, +4.66]e-4).
    # learning_rate and max_leaf_nodes were swept too and are already at their local optimum.
    cls = HistGradientBoostingClassifier if kind == "classify" else HistGradientBoostingRegressor
    return cls(
        max_iter=300, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=3.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        categorical_features=[c for c in CATEGORICAL if c in feature_cols], random_state=42,
    )


def fit_oof(sub: pd.DataFrame, y: np.ndarray, groups: np.ndarray, feature_cols: list[str], kind: str,
            totals: dict[str, pd.DataFrame] | None = None, n_splits: int = 5,
            fold_seeds: tuple[int, ...] | None = None) -> np.ndarray:
    """Out-of-fold predictions grouped by pitcher. With totals, the batter and catcher tendencies
    are rebuilt inside every fold.

    fold_seeds=None uses one deterministic grouped split. A tuple of seeds fits one random grouped split per
    seed and averages the predictions, each of which is out of fold on its own, so the average is too."""
    splitters = ([GroupKFold(n_splits=n_splits)] if fold_seeds is None
                 else [GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed) for seed in fold_seeds])
    return np.mean([fit_oof_once(sub, y, groups, feature_cols, kind, totals, splitter) for splitter in splitters], axis=0)


def fit_oof_once(sub: pd.DataFrame, y: np.ndarray, groups: np.ndarray, feature_cols: list[str], kind: str,
                 totals: dict[str, pd.DataFrame] | None, splitter: GroupKFold) -> np.ndarray:
    """One pass of out-of-fold predictions over the splitter's folds."""
    oof = np.full(len(sub), np.nan)
    # A fixed category set for season, decided once from the whole subset rather than per fold: a
    # fold's train and test splits must agree on which code means which year, and casting on `sub`
    # itself (instead of this copy) would risk changing how groupby("season") elsewhere in the
    # pipeline behaves on the shared, uncopied dataframe.
    season_dtype = pd.CategoricalDtype(sorted(sub["season"].unique())) if "season" in feature_cols else None
    for train_idx, test_idx in splitter.split(sub, y, groups=groups):
        train_pitchers = np.unique(groups[train_idx])
        train_rows, test_rows = sub.iloc[train_idx], sub.iloc[test_idx]
        X_train, X_test = train_rows[feature_cols].copy(), test_rows[feature_cols].copy()
        if season_dtype is not None:
            X_train["season"] = X_train["season"].astype(season_dtype)
            X_test["season"] = X_test["season"].astype(season_dtype)
        for name, tendency_totals in (totals or {}).items():
            key_col, split_by_hand = TENDENCY_KEYS[name]
            X_train[name] = tendency_for_rows(train_rows, tendency_totals, key_col, train_pitchers,
                                              leave_out_own=True, split_by_hand=split_by_hand)
            X_test[name] = tendency_for_rows(test_rows, tendency_totals, key_col, train_pitchers,
                                             leave_out_own=False, split_by_hand=split_by_hand)
        model = make_model(kind, feature_cols).fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1] if kind == "classify" else model.predict(X_test)
    return oof


def fit_tier(df: pd.DataFrame, out_col: str, target_col: str, subset_mask: pd.Series, kind: str,
             feature_cols: list[str], with_tendencies: bool = True, fold_seeds: tuple[int, ...] | None = None) -> None:
    """Writes out_col for every scored pitch type. Rare types join POOLED_WITH's model, or else share an OTHER model."""
    df[out_col] = np.nan
    outcome_rows = subset_mask & df["pitch_type"].notna()
    totals = ({name: build_totals(df, outcome_rows, key_col, target_col, split_by_hand=split_by_hand)
               for name, (key_col, split_by_hand) in TENDENCY_KEYS.items()}
              if with_tendencies else None)

    counts = df.loc[outcome_rows, "pitch_type"].value_counts()
    model_types = {t: [t] for t in counts.index if counts[t] >= MIN_N_FOR_MODEL}
    other = []
    for t in (t for t in counts.index if t not in model_types):
        if POOLED_WITH.get(t) in model_types:
            model_types[POOLED_WITH[t]].append(t)
        else:
            other.append(t)
    if other and counts[other].sum() >= MIN_N_FOR_MODEL // 2:
        model_types["OTHER"] = other

    for home, types in model_types.items():
        mask = outcome_rows & df["pitch_type"].isin(types)
        name = "+".join(types) if home != "OTHER" else "OTHER(" + "+".join(types) + ")"
        sub = df[mask]
        cols = feature_cols
        if len(types) > 1:
            sub = sub.assign(pitch_type=sub["pitch_type"].astype("category"))
            cols = feature_cols + ["pitch_type"]
        y = sub[target_col].to_numpy(float)
        pred = fit_oof(sub, y, sub["pitcher"].to_numpy(), cols, kind, totals, fold_seeds=fold_seeds)
        df.loc[mask, out_col] = pred
        score = f"AUC={roc_auc_score(y, pred):.3f}" if kind == "classify" else f"R2={r2_score(y, pred):.3f}"
        print(f"  {out_col}/{name}: n={mask.sum():,}, {score}", flush=True)


def fit_full_outcome(df: pd.DataFrame, label: str, target_col: str, subset_mask: pd.Series, kind: str,
                     extra_features: list[str] | None = None, fold_seeds: tuple[int, ...] | None = None) -> None:
    """Writes {label}_expected_full."""
    feature_cols = STUFF_FEATURES + LOCATION_FEATURES + (extra_features or []) + CONTEXT_FEATURES
    fit_tier(df, f"{label}_expected_full", target_col, subset_mask, kind, feature_cols, fold_seeds=fold_seeds)


def recalibrate_oof_isotonic(sub: pd.DataFrame, target_col: str, expected_col: str, n_splits: int = 5) -> np.ndarray:
    """Post-hoc monotonic recalibration of an out-of-fold continuous prediction: at the low end of
    timing's predicted range, actual deviation ran 0.24 in above what the model expected (the other
    9 deciles were within 0.08 in) — a real, if modest, miscalibration a squared-error regressor can
    leave at the tails. This is a second, independent out-of-fold step, grouped by pitcher exactly
    like the underlying model, so a pitcher's own actual outcomes never inform the curve used to
    recalibrate their own prediction."""
    groups = sub["pitcher"].to_numpy()
    x = sub[expected_col].to_numpy()
    y = sub[target_col].to_numpy()
    out = np.full(len(sub), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(sub, y, groups=groups):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x[train_idx], y[train_idx])
        out[test_idx] = iso.predict(x[test_idx])
    return out


def recalibrate_oof_by_group(sub: pd.DataFrame, target_col: str, expected_col: str, group_col: str,
                             binary: bool, n_splits: int = 5) -> np.ndarray:
    """Removes the league-wide level of each group (here year and month) from an out-of-fold expectation.

    Actual minus expected ran from -0.3 to +1.3 percentage points between months for whiff, and from
    -0.9 to +3.1 for ground ball, with no feature in the model that could see the date. A pitcher who
    throws mostly in one stretch of the season inherited that month's gap as if it were skill: the
    month alone moved a qualified pitcher by 2 to 3.5 index points (SD) per component and by 3 to 6 at
    the 95th percentile. Adding the date as a model feature barely moved the gap (four-seam whiff, SD
    0.54 to 0.51 percentage points), so this is a second out-of-fold step instead, grouped by pitcher
    like the models: each pitcher's offsets come only from other pitchers, so a pitcher's own outcomes
    never set the offset that scores them. A group the other pitchers never threw in gets no offset.
    Binary expectations stay inside [0, 1]."""
    groups = sub["pitcher"].to_numpy()
    expected = sub[expected_col].to_numpy(float)
    resid = sub[target_col].to_numpy(float) - expected
    keys = sub[group_col].to_numpy()
    out = np.full(len(sub), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(sub, resid, groups=groups):
        offset = pd.Series(resid[train_idx]).groupby(keys[train_idx]).mean()
        adjusted = expected[test_idx] + pd.Series(keys[test_idx]).map(offset).fillna(0.0).to_numpy()
        out[test_idx] = np.clip(adjusted, 0.0, 1.0) if binary else adjusted
    return out


def recalibrate_scored_by_month(df: pd.DataFrame, label: str, target_col: str, binary: bool) -> None:
    """Applies recalibrate_oof_by_group to every pitch that has a {label}_expected_full, in place."""
    col = f"{label}_expected_full"
    scored = df[col].notna()
    df.loc[scored, col] = recalibrate_oof_by_group(df.loc[scored], target_col, col, "year_month", binary)


if __name__ == "__main__":
    print("loading full pitch-level data...", flush=True)
    df = prepare_context(load_pitch_data(PITCH_LEVEL_FILE))
    existing = read_aligned_predictions(df)

    new_cols = []
    for label, spec in OUTCOMES.items():
        print(f"\n=== {label.upper()} (full tier: stuff+location+opponent+catcher+context) ===", flush=True)
        fit_full_outcome(df, label, spec["target"], df[spec["subset"]], spec["kind"], fold_seeds=FOLD_SEEDS)
        recalibrate_scored_by_month(df, label, spec["target"], binary=True)
        new_cols.append(f"{label}_expected_full")
    save_predictions(existing, df, new_cols)
