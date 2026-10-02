"""
Pitcher Deception Project: export site_stats.json
=================================================
Every number that appears in the page text (hero score, reliability, forecast
results, role gap, data cutoff) comes from here instead of being typed into the
HTML by hand. Hand-typed figures went stale several times during development;
generating them from the report CSVs means a rerun of the pipeline cannot leave
the pages contradicting themselves.

Run after add_timing_direction.py and predictive_validity.py, and before the two
build scripts. export_leaderboard_data.py imports pitcher_roles() from here.
"""

import json

import numpy as np
import pandas as pd

from reliability_and_ci import COMPOSITE_OUTCOMES, QUALIFY_MIN_N

# display order; membership in Deception+ comes from reliability_and_ci.COMPOSITE_OUTCOMES
COMPONENTS = [
    ("whiff", "Whiff"),
    ("chase", "Chase"),
    ("gb", "Ground ball"),
    ("weak", "Weak contact"),
    ("timing", "Timing"),
    ("whiffmiss", "Whiff miss distance"),
    ("calledstrike", "Called strike"),
    ("align", "Horizontal alignment"),
]
SP_MIN_PITCHES = 60
RP_MAX_PITCHES = 30
GS_SHARE_SP_MIN = 0.8
GS_SHARE_RP_MAX = 0.1


def fmt_r(x: float) -> str:
    return f"{x:.3f}"


def fmt_r_short(x: float) -> str:
    return f"{x:.3f}".lstrip("0")


def fmt_delta(x: float) -> str:
    return f"{x:+.3f}" if abs(x) >= 0.001 else f"{x:+.4f}"


def fmt_p(p: float) -> str:
    """Three decimals below 0.05, so a p of 0.012 next to a 0.01 gate does not read as 0.01."""
    if p < 1e-4:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    if p < 0.05:
        return f"{p:.3f}"
    return f"{p:.2f}"


QUALIFY_LABELS = {
    "whiff": "whiff", "chase": "chase", "gb": "ground ball", "weak": "weak contact",
    "timing": "timing", "whiffmiss": "whiff miss distance", "calledstrike": "called strike",
}


def join_words(words: list[str]) -> str:
    if len(words) <= 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + ", and " + words[-1]


def qualify_text() -> str:
    """'at least 100 in the ... samples and at least 40 in the ... samples', from QUALIFY_MIN_N."""
    by_min: dict[int, list[str]] = {}
    for label in COMPOSITE_OUTCOMES:
        by_min.setdefault(QUALIFY_MIN_N[label], []).append(QUALIFY_LABELS[label])
    parts = [f"at least {n} in the {join_words(words)} samples" for n, words in sorted(by_min.items(), reverse=True)]
    return " and ".join(parts)


def timing_vs_savant(ps: pd.DataFrame) -> float:
    """Pitcher-season correlation of the timing index with the share of on-time swings Savant's
    own swing-timing leaderboard reports, qualified pitchers only."""
    board = pd.read_csv("raw/swing_timing_by_pitcher_2025_2026.csv").rename(columns={"id": "pitcher"})
    joined = ps[ps["qualified"]].merge(board[["pitcher", "season", "on_time_percent"]], on=["pitcher", "season"])
    return float(joined["timing_index"].corr(joined["on_time_percent"]))


def stuff_plus_correlations(ps: pd.DataFrame) -> dict:
    """Correlation of Deception+ and each composite member's index with FanGraphs Stuff+, over qualified
    pitcher-seasons that have a Stuff+ figure: overall, by season, and per member. Stuff+ is only read
    here, never used in scoring."""
    q = ps[ps["qualified"] & ps["stuff_plus"].notna()]
    return {
        "n": len(q),
        "composite": float(q["deception_plus"].corr(q["stuff_plus"])),
        "by_season": {s: float(g["deception_plus"].corr(g["stuff_plus"])) for s, g in q.groupby("season")},
        "members": {k: float(q[f"{k}_index"].corr(q["stuff_plus"])) for k in COMPOSITE_OUTCOMES},
    }


def unscored_pitch_share(predictions_path: str = "output/per_pitch_predictions.csv") -> float:
    """Share of regular-season pitches left unscored: a pitch type with no model, a pitcher-season
    averaging under 75 mph, or missing core tracking all blank the pitch type."""
    return float(pd.read_csv(predictions_path, usecols=["pitch_type"])["pitch_type"].isna().mean())


def ground_ball_forecast_correlation(ps: pd.DataFrame) -> float:
    """Correlation of a pitcher's 2025 Deception+ with their 2026 ground-ball index."""
    a = ps[ps["season"] == 2025][["pitcher", "deception_plus"]].dropna()
    b = ps[ps["season"] == 2026][["pitcher", "gb_index"]].dropna()
    joined = a.merge(b, on="pitcher")
    return float(joined["deception_plus"].corr(joined["gb_index"]))


def clear_of_average_counts(ps: pd.DataFrame) -> tuple[int, int]:
    """(qualified Deception+ scores whose 95% interval excludes 100, qualified scores with an interval)."""
    q = ps[ps["qualified"] & ps["deception_plus_ci_lo"].notna() & ps["deception_plus_ci_hi"].notna()]
    clear = (q["deception_plus_ci_lo"] > 100) | (q["deception_plus_ci_hi"] < 100)
    return int(clear.sum()), len(q)


def sequencing_feature_overlap(driver: pd.DataFrame, ps: pd.DataFrame) -> dict:
    """How the two sequencing driver features relate to each other and to the whiff residual, over the
    pitcher-seasons the driver regression uses. Velocity gap averages the speed change over every
    consecutive pitch pair, repeats included, so it overlaps with repeat rate by construction."""
    cols = ["avg_velocity_gap_from_prev", "repeat_pct"]
    joined = driver.merge(ps[["pitcher", "season", "whiff_diff_adj_shrunk"]], on=["pitcher", "season"])
    joined = joined.dropna(subset=cols + ["whiff_diff_adj_shrunk"])
    return {
        "between": float(joined[cols[0]].corr(joined[cols[1]])),
        "gap_whiff": float(joined[cols[0]].corr(joined["whiff_diff_adj_shrunk"])),
        "repeat_whiff": float(joined[cols[1]].corr(joined["whiff_diff_adj_shrunk"])),
    }


def validation_text(validation: dict) -> dict:
    """Page-text figures from model_validation.json: how far each model's fit ranges across pitch types,
    and the calibration slopes."""
    outcomes = validation["outcomes"]

    def span(label: str, metric: str) -> str:
        values = [r[metric] for r in outcomes[label]["by_pitch_type"]]
        return f"{min(values):.2f} to {max(values):.2f}"

    member_slopes = [outcomes[k]["calibration"]["slope"] for k in COMPOSITE_OUTCOMES]
    return {
        "VAL_AUC_GB": span("gb", "auc"),
        "VAL_AUC_CALLEDSTRIKE": span("calledstrike", "auc"),
        "VAL_R2_TIMING": span("timing", "r2"),
        "VAL_R2_WHIFFMISS": span("whiffmiss", "r2"),
        "VAL_SLOPE_MEMBERS": f"{min(member_slopes):.2f} to {max(member_slopes):.2f}",
        "VAL_SLOPE_GB": f"{outcomes['gb']['calibration']['slope']:.2f}",
    }


def component_yoy(ps: pd.DataFrame, key: str) -> float:
    """Correlation of a component's index across seasons, both seasons held to the same qualifying
    minimum. Used to halve the 2026 threshold from when 2026 was a partial season; both seasons are
    now complete, and the asymmetry was measurably diluting several of these correlations (+0.004 to
    +0.010 once matched) by pairing a fully-qualified 2025 side against a noisier, under-qualified
    2026 side."""
    min_n = QUALIFY_MIN_N[key]
    cols = ["pitcher", f"{key}_index", f"{key}_n"]
    a = ps[ps["season"] == 2025][cols].dropna()
    b = ps[ps["season"] == 2026][cols].dropna()
    joined = a[a[f"{key}_n"] >= min_n].merge(b[b[f"{key}_n"] >= min_n], on="pitcher", suffixes=("_25", "_26"))
    return float(joined[f"{key}_index_25"].corr(joined[f"{key}_index_26"]))


def pretty_name(name: str) -> str:
    return " ".join(reversed(name.split(", ")))


def role_label(median_pitches: float) -> str:
    if median_pitches >= SP_MIN_PITCHES:
        return "SP"
    if median_pitches < RP_MAX_PITCHES:
        return "RP"
    return "MR"


def real_role_label(gs_share: float) -> str | float:
    if pd.isna(gs_share):
        return np.nan
    if gs_share >= GS_SHARE_SP_MIN:
        return "SP"
    if gs_share < GS_SHARE_RP_MAX:
        return "RP"
    return "MR"


def pitcher_roles(raw: pd.DataFrame, usage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Role label: SP/RP/MR per pitcher-season. Prefers the real games-started share (usage: pitcher,
    season, games, games_started, from the FanGraphs standard export) over the pitches-per-appearance
    proxy alone — the proxy mislabels 17.7% of pitcher-seasons on 2025-26 data, concentrated in
    swingmen and spot starters a single pitch-count threshold can't tell apart from a true starter or
    reliever. Falls back to the proxy where usage is missing or not passed, so this still works
    without that manual export. raw needs pitcher, season, game_pk."""
    per_game = raw.groupby(["pitcher", "season", "game_pk"]).size().reset_index(name="pitches")
    roles = per_game.groupby(["pitcher", "season"])["pitches"].median().rename("med_pitches_per_app").reset_index()
    proxy_role = roles["med_pitches_per_app"].map(role_label)

    real_role = pd.Series(np.nan, index=roles.index, dtype=object)
    if usage is not None:
        keyed = roles[["pitcher", "season"]].merge(
            usage[["pitcher", "season", "games", "games_started"]], on=["pitcher", "season"], how="left")
        gs_share = np.where(keyed["games"] > 0, keyed["games_started"] / keyed["games"], np.nan)
        real_role = pd.Series(gs_share, index=roles.index).map(real_role_label)

    roles["role"] = real_role.where(real_role.notna(), proxy_role)
    return roles


def main() -> None:
    ps = pd.read_csv("output/pitcher_season.csv")
    rr = pd.read_csv("output/reliability_report.csv").set_index("label")
    pv = pd.read_csv("output/predictive_validity_report.csv").set_index("label")

    raw = pd.read_csv(
        "raw/statcast_pitch_level_2025_2026.csv",
        usecols=["pitcher", "season", "game_pk", "game_date"],
        low_memory=False,
    )
    data_through = pd.to_datetime(raw["game_date"]).max()

    ps = ps.merge(pitcher_roles(raw, usage=ps[["pitcher", "season", "games", "games_started"]]),
                  on=["pitcher", "season"], how="left")
    qualified = ps[ps["qualified"]].copy()

    # year over year, same definition as reliability_and_ci.py
    scored = ps.dropna(subset=["deception_plus"])
    yoy = scored[scored["season"] == 2025].merge(
        scored[scored["season"] == 2026], on="pitcher", suffixes=("_25", "_26")
    )
    yoy_r = yoy["deception_plus_25"].corr(yoy["deception_plus_26"])

    # share of variance on the first principal component across the composite members
    z_cols = [f"{k}_z" for k in COMPOSITE_OUTCOMES]
    valid = ps.dropna(subset=z_cols)
    eigvals = np.linalg.eigvalsh(valid[z_cols].corr().values)
    pc1 = eigvals.max() / len(z_cols)

    rel = {k: float(rr.loc[k, "r_full_spearman_brown"]) for k, _ in COMPONENTS}
    member_rel = {k: rel[k] for k in COMPOSITE_OUTCOMES}
    most = max(member_rel, key=member_rel.get)
    least = min(member_rel, key=member_rel.get)
    yoy_by_component = {k: component_yoy(ps, k) for k, _ in COMPONENTS}
    design_effects = [float(rr.loc[k, "design_effect"]) for k, _ in COMPONENTS]
    label_of = dict(COMPONENTS)

    stuff = stuff_plus_correlations(ps)
    sequencing = sequencing_feature_overlap(pd.read_csv("output/driver_features.csv"), ps)
    clear = clear_of_average_counts(ps)

    top = qualified.sort_values("deception_plus", ascending=False).head(25)
    top2 = top.head(2)

    def pitcher_row(name: str, season: int) -> pd.Series:
        row = ps[(ps["player_name"] == name) & (ps["season"] == season)]
        return row.iloc[0]

    rogers = pitcher_row("Rogers, Tyler", 2025)
    skubal = pitcher_row("Skubal, Tarik", 2025)

    components = []
    for key, label in COMPONENTS:
        components.append({
            "key": key,
            "label": label,
            "reliability": fmt_r(rel[key]),
            "yoy": fmt_r(yoy_by_component[key]),
            "delta": fmt_delta(float(pv.loc[key, "cv_delta_r2"])),
            "p": fmt_p(float(pv.loc[key, "f_pvalue"])),
            "n": int(pv.loc[key, "n"]),
            "passes": bool(pv.loc[key, "f_pvalue"] < 0.01),
            "in_score": key in COMPOSITE_OUTCOMES,
        })
    components.append({
        "key": "composite",
        "label": "Deception+ (forecast is for 2026 whiff rate)",
        "reliability": fmt_r(float(rr.loc["deception_plus", "r_full_spearman_brown"])),
        "yoy": fmt_r(yoy_r),
        "in_score": True,
        "delta": fmt_delta(float(pv.loc["composite_to_whiff", "cv_delta_r2"])),
        "p": fmt_p(float(pv.loc["composite_to_whiff", "f_pvalue"])),
        "n": int(pv.loc["composite_to_whiff", "n"]),
        "passes": bool(pv.loc["composite_to_whiff", "f_pvalue"] < 0.01),
    })

    with open("output/artifact_data.json", encoding="utf-8") as f:
        n_points = len(json.load(f)["points"])
    with open("output/driver_analysis.json", encoding="utf-8") as f:
        driver_r2 = [m["r2_mean"] for m in json.load(f).values()]
    with open("output/sequencing_driver_report.json", encoding="utf-8") as f:
        seq_r2 = [m["r2"] for m in json.load(f).values()]
    with open("output/model_validation.json", encoding="utf-8") as f:
        validation = json.load(f)

    text = {
        "DATA_THROUGH": f"{data_through:%B} {data_through.day}, {data_through.year}",
        "N_SCORED": f"{int(ps['deception_plus'].notna().sum()):,}",
        "N_QUALIFIED": f"{int(ps['qualified'].sum()):,}",
        "N_POINTS": f"{n_points:,}",
        "COMPOSITE_R": fmt_r(float(rr.loc["deception_plus", "r_full_spearman_brown"])),
        "YOY_R": fmt_r(yoy_r),
        "YOY_N": f"{len(yoy):,}",
        "PC1": f"{pc1 * 100:.1f}%",
        "MOST_LEAST": f"{label_of[most].split()[0].lower()} {fmt_r_short(rel[most])} / "
                      f"{label_of[least].split()[0].lower()} {fmt_r_short(rel[least])}",
        "LEAST_LABEL": label_of[least].lower(),
        "LEAST_R": fmt_r(rel[least]),
        "DE_RANGE": f"{min(design_effects):.2f} to {max(design_effects):.2f}",
        "TOP1_NAME": pretty_name(top2.iloc[0]["player_name"]),
        "TOP1_SEASON": str(int(top2.iloc[0]["season"])),
        "TOP1_SCORE": f"{top2.iloc[0]['deception_plus']:.1f}",
        "TOP2_NAME": pretty_name(top2.iloc[1]["player_name"]),
        "TOP2_SEASON": str(int(top2.iloc[1]["season"])),
        "TOP2_SCORE": f"{top2.iloc[1]['deception_plus']:.1f}",
        "ROLE_SP_MIN": str(SP_MIN_PITCHES),
        "ROLE_RP_MAX": str(RP_MAX_PITCHES),
        "ROLE_GS_MIN": f"{GS_SHARE_SP_MIN * 100:.0f}%",
        "ROLE_GS_MAX": f"{GS_SHARE_RP_MAX * 100:.0f}%",
        "SP_MEAN": f"{qualified.loc[qualified['role'] == 'SP', 'deception_plus'].mean():.1f}",
        "RP_MEAN": f"{qualified.loc[qualified['role'] == 'RP', 'deception_plus'].mean():.1f}",
        "RP_TOP25": str(int((top["role"] == "RP").sum())),
        "ROGERS_WHIFF": f"{rogers['whiff_index']:.0f}",
        "ROGERS_GB": f"{rogers['gb_index']:.0f}",
        "ROGERS_DP": f"{rogers['deception_plus']:.0f}",
        "ROGERS_TIMING_DIR": f"{abs(rogers['timing_bias_inches']):.1f}",
        "ROGERS_TIMING_WORD": "late" if rogers["timing_bias_inches"] < 0 else "early",
        "SKUBAL_WHIFF": f"{skubal['whiff_index']:.0f}",
        "SKUBAL_GB": f"{skubal['gb_index']:.0f}",
        "SKUBAL_DP": f"{skubal['deception_plus']:.0f}",
        "REL_CALLEDSTRIKE": fmt_r(rel["calledstrike"]),
        "REL_WHIFF": fmt_r(rel["whiff"]),
        "GB_CORR_MAX": fmt_r(max(abs(float(qualified["gb_index"].corr(qualified[f"{m}_index"])))
                                 for m in COMPOSITE_OUTCOMES if m != "gb")),
        "MOST_LABEL": label_of[most].lower(),
        "MOST_R": fmt_r(rel[most]),
        "CORR_WHIFF_TIMING": fmt_r(float(qualified["whiff_index"].corr(qualified["timing_index"]))),
        "DELTA_TIMING": fmt_delta(float(pv.loc["timing", "cv_delta_r2"])),
        "P_CALLEDSTRIKE": fmt_p(float(pv.loc["calledstrike", "f_pvalue"])),
        "YOY_ALIGN": fmt_r(yoy_by_component["align"]),
        "YOY_CALLEDSTRIKE": fmt_r(yoy_by_component["calledstrike"]),
        "P_ALIGN": fmt_p(float(pv.loc["align", "f_pvalue"])),
        "DELTA_ALIGN": fmt_delta(float(pv.loc["align", "cv_delta_r2"])),
        "N_COMPOSITE": str(len(COMPOSITE_OUTCOMES)),
        "QUALIFY_TEXT": qualify_text(),
        "TIMING_ONTIME_R": fmt_r(timing_vs_savant(ps)),
        "REL_CHASE": fmt_r(rel["chase"]),
        "DRIVER_R2_RANGE": f"{min(driver_r2):.2f} to {max(driver_r2):.2f}",
        "SEQ_R2_RANGE": f"{min(seq_r2):.4f} to {max(seq_r2):.4f}",
        "STUFF_R": fmt_r(stuff["composite"]),
        "STUFF_R_2025": fmt_r(stuff["by_season"][2025]),
        "STUFF_R_2026": fmt_r(stuff["by_season"][2026]),
        "STUFF_N": f"{stuff['n']:,}",
        "STUFF_R_MEMBERS": join_words([f"{QUALIFY_LABELS[k]} {fmt_r_short(r)}" for k, r in stuff["members"].items()]),
        "DELTA_CALLEDSTRIKE": fmt_delta(float(pv.loc["calledstrike", "cv_delta_r2"])),
        "UNSCORED_PCT": f"{unscored_pitch_share() * 100:.1f}%",
        "GB_FORECAST_R": fmt_r(ground_ball_forecast_correlation(ps)),
        "DP_HALF": f"{((qualified['deception_plus_ci_hi'] - qualified['deception_plus_ci_lo']) / 2).median():.0f}",
        "DP_CLEAR": f"{clear[0]:,}",
        "DP_CLEAR_PCT": f"{clear[0] / clear[1]:.0%}",
        "SEQ_FEATURES_R": fmt_r(sequencing["between"]),
        "SEQ_GAP_WHIFF_R": fmt_r(sequencing["gap_whiff"]),
        "SEQ_REPEAT_WHIFF_R": fmt_r(sequencing["repeat_whiff"]),
        **validation_text(validation),
    }

    out = {"text": text, "components": components}
    with open("output/site_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved site_stats.json ({len(text)} text fields, {len(components)} table rows).")
    print(f"data through {text['DATA_THROUGH']}; reliever mean {text['RP_MEAN']} vs starter mean {text['SP_MEAN']}; "
          f"{text['RP_TOP25']} of top 25 qualified are relievers.")


if __name__ == "__main__":
    main()
