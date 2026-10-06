"""
Pitcher Deception Project: candidate driver features and the driver regression
==============================================================================
The scored residuals answer "who beats their pitch traits". This module builds
the season-level pitcher traits that might explain why, and fits the driver
regression that tests them (driver_analysis.py runs it).

Features, one row per pitcher-season:
  release-point, arm-angle and VAA spread across a pitcher's pitch types
  velocity gap when the pitch type switches from the previous pitch, same-pitch-type repeat rate
  release extension, average VAA, pitch-mix entropy, pace, arm angle, repertoire size
  tunnel differential: how much closer two consecutive pitches of different
    types are at the batter's decision point (23.5 ft out) than at the plate
  spin-axis mirror score: circular distance between consecutive pitches' spin
    axes, folded so a perfect match and a perfect mirror both score 0

Deliberately left out:
  Savant's swing-outcome stats (miss distance, tied up %, flailed %, early %,
  late %). They are outcomes of the same swings the residual scores, so
  regressing the residual on them would count one signal twice.
  The tempo pull's second pace column. It is a byte-for-byte copy of the first,
  and Ridge split the coefficient exactly in half between the two.

Run after merge_data.py. Reads pitcher_pitchtype_season.csv,
pitcher_season_covariates.csv and the pitch-level file. Writes driver_features.csv.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

PITCH_LEVEL_FILE = "raw/statcast_pitch_level_2024_2026.csv"
CHUNKSIZE = 500_000
MIN_PITCHES_PER_TYPE = 20
Y0 = 50.0            # Statcast trajectory start, ft from home plate
Y_PLATE = 17 / 12    # front of home plate, ft
Y_TUNNEL = 23.5      # approximate batter decision point, ft from home plate

SEQUENCE_KEYS = ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
TUNNEL_COLS = SEQUENCE_KEYS + ["pitch_type", "season", "plate_x", "plate_z",
                                "vx0", "vy0", "vz0", "ax", "ay", "az"]
SPIN_COLS = SEQUENCE_KEYS + ["pitch_type", "season", "spin_axis"]
COVARIATE_COLS = ["pitcher", "season", "pitch_mix_entropy", "tempo_bases_empty_sec", "arm_angle_szn_avg"]

FEATURE_LABELS = {
    "spin_mirror_score_mean": "Spin-axis mirror score (real; low = match/mirror)",
    "tunnel_differential_mean": "Pitch-pair tunnel differential (real)",
    "arm_angle_cross_pitch_std": "Arm-angle consistency (season proxy)",
    "release_pos_x_cross_pitch_std": "Release-point (x) consistency (season proxy)",
    "release_pos_z_cross_pitch_std": "Release-point (z) consistency (season proxy)",
    "vaa_cross_pitch_std": "VAA consistency (season proxy)",
    "velocity_gap_per_switch": "Velocity gap when switching pitch types",
    "repeat_pct": "Same-pitch-type repeat rate",
    "release_extension_mean": "Release extension",
    "vaa_mean": "Average vertical approach angle",
    "pitch_mix_entropy": "Pitch-mix unpredictability (entropy)",
    "tempo_bases_empty_sec": "Pace between pitches",
    "arm_angle_szn_avg": "Season-average arm angle",
    "n_pitch_types": "Repertoire size (# pitch types)",
}


def load_columns(path: str, cols: list[str]) -> pd.DataFrame:
    """The requested columns for regular-season pitches, like the scores. Spring training and the postseason
    are 3.9% and 1.9% of 2025 pitches and would otherwise leak into the season-level features."""
    chunks = pd.read_csv(path, usecols=list(cols) + ["game_type"], chunksize=CHUNKSIZE, low_memory=False)
    frame = pd.concat(chunks, ignore_index=True)
    return frame[frame["game_type"] == "R"].drop(columns="game_type").reset_index(drop=True)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    avg = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - avg) ** 2, weights=weights)))


def pooled_gap_per_switch(pitch_types: pd.DataFrame) -> float:
    """Average speed change on the pitches that switched type, over all of a pitcher-season's pitch types, each
    weighted by its switches. NaN when none of them has one."""
    has_switch = pitch_types["avg_velocity_gap_per_switch"].notna() & (pitch_types["n_switched"] > 0)
    if not has_switch.any():
        return np.nan
    return float(np.average(pitch_types.loc[has_switch, "avg_velocity_gap_per_switch"], weights=pitch_types.loc[has_switch, "n_switched"]))


def build_cross_pitch_features(pitch_types: pd.DataFrame) -> pd.DataFrame:
    """Spread of release point, arm angle and VAA across a pitcher's pitch types."""
    rows = []
    for (pitcher, season), g in pitch_types[pitch_types["pitches"] >= MIN_PITCHES_PER_TYPE].groupby(["pitcher", "season"]):
        if len(g) < 2:
            continue
        weights = g["pitches"]
        rows.append({
            "pitcher": pitcher, "season": season, "n_pitch_types": len(g),
            "arm_angle_cross_pitch_std": weighted_std(g["arm_angle_mean"].values, weights.values),
            "release_pos_x_cross_pitch_std": weighted_std(g["release_pos_x_mean"].values, weights.values),
            "release_pos_z_cross_pitch_std": weighted_std(g["release_pos_z_mean"].values, weights.values),
            "vaa_cross_pitch_std": weighted_std(g["vaa_mean"].values, weights.values),
            "velocity_gap_per_switch": pooled_gap_per_switch(g),
            "repeat_pct": np.average(g["repeat_pct"].fillna(g["repeat_pct"].mean()), weights=weights),
            "release_extension_mean": np.average(g["release_extension_mean"], weights=weights),
            "vaa_mean": np.average(g["vaa_mean"], weights=weights),
        })
    return pd.DataFrame(rows)


def traj_xy_at(plate_x, plate_z, vx0, vy0, vz0, ax, ay, az, y_target):
    """Position (x, z) at distance y_target from home plate, worked back from the
    known plate crossing with the constant-acceleration model Statcast publishes."""
    vy_plate = -np.sqrt(np.maximum(vy0 ** 2 - 2 * ay * (Y0 - Y_PLATE), 0))
    t_plate = (vy_plate - vy0) / ay
    x0 = plate_x - vx0 * t_plate - 0.5 * ax * t_plate ** 2
    z0 = plate_z - vz0 * t_plate - 0.5 * az * t_plate ** 2

    vy_t = -np.sqrt(np.maximum(vy0 ** 2 - 2 * ay * (Y0 - y_target), 0))
    t_t = (vy_t - vy0) / ay
    x_t = x0 + vx0 * t_t + 0.5 * ax * t_t ** 2
    z_t = z0 + vz0 * t_t + 0.5 * az * t_t ** 2
    return x_t, z_t


def compute_tunnel_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Tunnel-point position of each pitch, paired with the previous pitch in the
    same at-bat. Cross-pitch-type pairs only. Returns a pitcher-season aggregate."""
    x_t, z_t = traj_xy_at(
        df["plate_x"].values, df["plate_z"].values,
        df["vx0"].values, df["vy0"].values, df["vz0"].values,
        df["ax"].values, df["ay"].values, df["az"].values, Y_TUNNEL,
    )
    seq = df[SEQUENCE_KEYS + ["pitch_type", "season", "plate_x", "plate_z"]].copy()
    seq["x_tunnel"] = x_t
    seq["z_tunnel"] = z_t
    seq = seq.sort_values(SEQUENCE_KEYS)

    grp = seq.groupby(["pitcher", "game_pk", "at_bat_number"])
    for col in ["pitch_type", "plate_x", "plate_z", "x_tunnel", "z_tunnel"]:
        seq[f"prev_{col}"] = grp[col].shift(1)

    valid = seq.dropna(subset=["prev_pitch_type"]).copy()
    valid = valid[valid["pitch_type"] != valid["prev_pitch_type"]]

    plate_dist = np.sqrt((valid["plate_x"] - valid["prev_plate_x"]) ** 2 +
                         (valid["plate_z"] - valid["prev_plate_z"]) ** 2) * 12
    tunnel_dist = np.sqrt((valid["x_tunnel"] - valid["prev_x_tunnel"]) ** 2 +
                          (valid["z_tunnel"] - valid["prev_z_tunnel"]) ** 2) * 12
    valid["tunnel_differential_in"] = plate_dist - tunnel_dist

    return valid.groupby(["pitcher", "season"]).agg(
        tunnel_differential_mean=("tunnel_differential_in", "mean"),
        tunnel_pairs_n=("tunnel_differential_in", "size"),
    ).reset_index()


def circular_mirror_score(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0 for a matching or perfectly mirrored spin axis pair, 90 for perpendicular.
    Casts explicitly and returns NaN when either input is NaN, so callers need not
    pre-filter."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = np.abs(a - b) % 360
    diff = np.minimum(diff, 360 - diff)          # 0-180, circular distance
    out = np.minimum(diff, np.abs(180 - diff))   # 0-90, low = match OR mirror
    return np.where(np.isnan(a) | np.isnan(b), np.nan, out)


def compute_spin_mirror(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["spin_axis"]).sort_values(SEQUENCE_KEYS)
    grp = df.groupby(["pitcher", "game_pk", "at_bat_number"])
    df["prev_pitch_type"] = grp["pitch_type"].shift(1)
    df["prev_spin_axis"] = grp["spin_axis"].shift(1)

    valid = df.dropna(subset=["prev_pitch_type"]).copy()
    valid = valid[valid["pitch_type"] != valid["prev_pitch_type"]]
    valid["spin_mirror_score"] = circular_mirror_score(valid["spin_axis"].values, valid["prev_spin_axis"].values)

    return valid.groupby(["pitcher", "season"]).agg(
        spin_mirror_score_mean=("spin_mirror_score", "mean"),
        spin_mirror_pairs_n=("spin_mirror_score", "size"),
    ).reset_index()


def build_driver_features(pitch_types: pd.DataFrame, covariates: pd.DataFrame,
                          tunnel: pd.DataFrame, spin: pd.DataFrame) -> pd.DataFrame:
    features = build_cross_pitch_features(pitch_types)
    features = features.merge(covariates[COVARIATE_COLS], on=["pitcher", "season"], how="left")
    features = features.merge(tunnel.drop(columns=["tunnel_pairs_n"]), on=["pitcher", "season"], how="left")
    return features.merge(spin.drop(columns=["spin_mirror_pairs_n"]), on=["pitcher", "season"], how="left")


def fdr_adjusted_p_values(X: pd.DataFrame, y: pd.Series, groups: np.ndarray | None = None) -> pd.Series:
    """Per-feature p-values from a multiple OLS fit, adjusted across the features
    (Benjamini-Hochberg). With 14 features per outcome, about one raw p
    below 0.05 is expected by chance alone, so the adjusted value is the one to read.

    `groups` clusters the standard errors, one cluster per pitcher. A pitcher with both seasons in
    the sample contributes two rows whose residuals and features are strongly dependent (the
    same-pitcher correlation of the whiff residual across seasons is 0.49), so classical errors
    overstate the evidence."""
    model = sm.OLS(y, sm.add_constant(X))
    fit = model.fit() if groups is None else model.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
    p = fit.pvalues[X.columns]
    return pd.Series(multipletests(p.values, method="fdr_bh")[1], index=X.columns)


def run_driver_analysis(driver_df: pd.DataFrame, scores: pd.DataFrame, target_col: str) -> dict:
    """Random forest (grouped CV R^2, permutation importance) plus Ridge coefficients
    of a residual score on the driver features. Splits are grouped by pitcher, since
    most pitchers have both a 2025 and a 2026 row and a row-level split would let a
    pitcher's other season leak into the fold that scores him."""
    data = driver_df.merge(scores[["pitcher", "season", target_col]], on=["pitcher", "season"], how="inner")
    data = data.dropna(subset=[target_col])
    feature_cols = [c for c in driver_df.columns if c not in ("pitcher", "season")]
    data = data.dropna(subset=feature_cols, thresh=len(feature_cols) - 2)
    for c in feature_cols:
        data[c] = data[c].fillna(data[c].median())

    X, y = data[feature_cols], data[target_col]
    pitcher_groups = data["pitcher"].values

    rf = RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1)
    cv_r2 = cross_val_score(rf, X, y, cv=GroupKFold(n_splits=5), groups=pitcher_groups, scoring="r2")

    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, pitcher_groups))
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    perm = permutation_importance(rf, X.iloc[test_idx], y.iloc[test_idx], n_repeats=30, random_state=42, n_jobs=-1)

    ridge = Ridge(alpha=1.0)
    ridge.fit(StandardScaler().fit_transform(X), y)

    q = fdr_adjusted_p_values(X, y, groups=pitcher_groups)
    rows = [{"feature": FEATURE_LABELS.get(feat, feat),
             "importance": round(float(perm.importances_mean[i]), 5),
             "coef": round(float(ridge.coef_[i]), 5),
             "q": round(float(q[feat]), 5)}
            for i, feat in enumerate(feature_cols)]
    rows.sort(key=lambda r: -r["importance"])
    result = {"n": len(data), "r2_mean": round(float(cv_r2.mean()), 3),
              "r2_std": round(float(cv_r2.std()), 3), "features": rows}
    print(f"\nDriver analysis on {target_col}: n={len(data)}, R^2 mean={result['r2_mean']} (std={result['r2_std']})")
    print(f"  {int((q < 0.05).sum())} of {len(feature_cols)} features survive the Benjamini-Hochberg correction (q < 0.05)")
    for r in rows[:8]:
        print(f"  {r['feature']:50s} importance={r['importance']:.4f} coef={r['coef']:+.5f} q={r['q']:.4f}")
    return result


if __name__ == "__main__":
    print("computing tunnel differential over cross-type pitch pairs...")
    tunnel = compute_tunnel_differential(load_columns(PITCH_LEVEL_FILE, TUNNEL_COLS))
    print(f"scored {len(tunnel):,} pitcher-seasons, mean tunnel differential = "
          f"{tunnel['tunnel_differential_mean'].mean():.2f} in")

    print("computing spin-axis mirror score over the same pairs...")
    spin = compute_spin_mirror(load_columns(PITCH_LEVEL_FILE, SPIN_COLS))
    print(f"scored {len(spin):,} pitcher-seasons, mean mirror score = "
          f"{spin['spin_mirror_score_mean'].mean():.1f} deg (0 = match or mirror, 90 = perpendicular)")

    features = build_driver_features(
        pd.read_csv("output/pitcher_pitchtype_season.csv"),
        pd.read_csv("output/pitcher_season_covariates.csv"),
        tunnel, spin,
    )
    features.to_csv("output/driver_features.csv", index=False)
    print(f"\nSaved driver_features.csv: {len(features):,} pitcher-seasons, "
          f"{len(features.columns) - 2} features. Done.")
