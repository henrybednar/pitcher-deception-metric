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
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from driver_features import FEATURE_LABELS, MIN_PITCHES_PER_TYPE, weighted_std
from reliability_and_ci import COMPOSITE_OUTCOMES, QUALIFY_MIN_N
from season_pairs import cluster_bootstrap_interval, consecutive_pairs, correlation_by_pair, pooled_correlation

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
    board = pd.read_csv("raw/swing_timing_by_pitcher_2024_2026.csv").rename(columns={"id": "pitcher"})
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


BOOTSTRAP_DRAWS = 1000


def year_over_year_summary(ps: pd.DataFrame, n_boot: int = BOOTSTRAP_DRAWS, seed: int = 0) -> dict:
    """Year-over-year correlation of Deception+ among pitchers qualified in both seasons of a pair, pooled over
    every pair of consecutive seasons: the figure, a 95% interval from resampling pitchers (a pitcher's two pairs
    share a season), the figure for each pair, and the figure for the higher- and lower-volume halves. A pair's
    volume is the smaller of its two seasons' mean member sample size relative to that member's median. The
    headline tile uses every scored pitcher-season instead, which is lower because small samples are noisier."""
    members = COMPOSITE_OUTCOMES
    q = ps[ps["qualified"] & ps["deception_plus"].notna()].copy()
    medians = {m: q[f"{m}_n"].median() for m in members}
    q["volume"] = sum(q[f"{m}_n"] / medians[m] for m in members) / len(members)
    pairs = consecutive_pairs(q, ["deception_plus", "volume"])
    first, second = pairs["deception_plus_1"].to_numpy(), pairs["deception_plus_2"].to_numpy()
    volume = pairs[["volume_1", "volume_2"]].min(axis=1).to_numpy()
    lo, hi = cluster_bootstrap_interval(first, second, pairs["pitcher"].to_numpy(), n_boot, seed)
    high = volume >= np.median(volume)
    return {"r": float(np.corrcoef(first, second)[0, 1]), "n": int(len(pairs)), "lo": lo, "hi": hi,
            "high_volume_r": float(np.corrcoef(first[high], second[high])[0, 1]),
            "low_volume_r": float(np.corrcoef(first[~high], second[~high])[0, 1]),
            "by_pair": correlation_by_pair(pairs, "deception_plus")}


def stuff_plus_correlation_interval(ps: pd.DataFrame, n_boot: int = BOOTSTRAP_DRAWS, seed: int = 0) -> tuple[float, float]:
    """95% interval for the Deception+ and Stuff+ correlation over qualified pitcher-seasons, resampling whole
    pitchers: most pitchers contribute more than one season, so resampling rows would understate the uncertainty."""
    q = ps[ps["qualified"] & ps["stuff_plus"].notna() & ps["deception_plus"].notna()]
    return cluster_bootstrap_interval(q["stuff_plus"].to_numpy(), q["deception_plus"].to_numpy(), q["pitcher"].to_numpy(), n_boot, seed)


def handedness_gap(ps: pd.DataFrame, hands: pd.Series) -> dict:
    """Left-handed minus right-handed mean Deception+ among qualified pitcher-seasons, with a p-value whose errors
    cluster on pitcher (a pitcher contributes up to three seasons, so rows are not independent draws). The gap is
    the coefficient of a regression on a left-hander indicator, which is exactly the difference in row means.
    Handedness is a model input, so a gap is either real or a leftover miscalibration."""
    q = ps[ps["qualified"] & ps["deception_plus"].notna()]
    hand = q["pitcher"].map(hands)
    q, hand = q[hand.isin(["L", "R"])], hand[hand.isin(["L", "R"])]
    left = (hand == "L").astype(float).to_numpy()
    fit = sm.OLS(q["deception_plus"].to_numpy(), sm.add_constant(left)).fit(cov_type="cluster", cov_kwds={"groups": q["pitcher"].to_numpy()})
    return {"gap": float(fit.params[1]), "p": float(fit.pvalues[1]),
            "n_left": int((hand == "L").sum()), "n_right": int((hand == "R").sum())}


def unscored_pitch_share(predictions_path: str = "output/per_pitch_predictions.csv") -> float:
    """Share of regular-season pitches left unscored: a pitch type with no model, a pitcher-season
    averaging under 75 mph, or missing core tracking all blank the pitch type."""
    return float(pd.read_csv(predictions_path, usecols=["pitch_type"])["pitch_type"].isna().mean())


def ground_ball_forecast_correlation(ps: pd.DataFrame) -> float:
    """Correlation of a pitcher's Deception+ with their ground-ball index the next season, over every pair of
    consecutive seasons."""
    earlier = ps[["pitcher", "season", "deception_plus"]].dropna()
    later = ps[["pitcher", "season", "gb_index"]].dropna().assign(season=lambda d: d["season"] - 1)
    joined = earlier.merge(later, on=["pitcher", "season"])
    return float(joined["deception_plus"].corr(joined["gb_index"]))


def clear_of_average_counts(ps: pd.DataFrame) -> tuple[int, int]:
    """(qualified Deception+ scores whose 95% interval excludes 100, qualified scores with an interval)."""
    q = ps[ps["qualified"] & ps["deception_plus_ci_lo"].notna() & ps["deception_plus_ci_hi"].notna()]
    clear = (q["deception_plus_ci_lo"] > 100) | (q["deception_plus_ci_hi"] < 100)
    return int(clear.sum()), len(q)


def sequencing_feature_overlap(driver: pd.DataFrame, ps: pd.DataFrame) -> dict:
    """How the two sequencing driver features relate to each other and to the whiff residual, over the
    pitcher-seasons the driver regression uses. The velocity gap is taken on pitch-type switches only, so it
    no longer carries the repeat rate."""
    cols = ["velocity_gap_per_switch", "repeat_pct"]
    joined = driver.merge(ps[["pitcher", "season", "whiff_diff_adj_shrunk"]], on=["pitcher", "season"])
    joined = joined.dropna(subset=cols + ["whiff_diff_adj_shrunk"])
    return {
        "between": float(joined[cols[0]].corr(joined[cols[1]])),
        "gap_whiff": float(joined[cols[0]].corr(joined["whiff_diff_adj_shrunk"])),
        "repeat_whiff": float(joined[cols[1]].corr(joined["whiff_diff_adj_shrunk"])),
    }


def speed_spread_correlation(driver: pd.DataFrame, pitch_types: pd.DataFrame) -> float:
    """Correlation of the velocity gap per switch with how far apart a pitcher's pitch types sit in speed (the
    pitch-weighted SD of their average speeds), over the pitcher-seasons that have both."""
    rows = [{"pitcher": pitcher, "season": season, "speed_spread": weighted_std(g["release_speed_mean"].to_numpy(), g["pitches"].to_numpy())}
            for (pitcher, season), g in pitch_types[pitch_types["pitches"] >= MIN_PITCHES_PER_TYPE].groupby(["pitcher", "season"])]
    joined = driver.merge(pd.DataFrame(rows), on=["pitcher", "season"]).dropna(subset=["velocity_gap_per_switch", "speed_spread"])
    return float(joined["velocity_gap_per_switch"].corr(joined["speed_spread"]))


def sequencing_survivors(driver_analysis: dict) -> dict:
    """Page-text lists of the outcomes for which each sequencing feature survives the Benjamini-Hochberg correction
    (q below 0.05) in driver_analysis.json."""
    labels = {**QUALIFY_LABELS, "align": "horizontal alignment"}

    def survives(feature: str) -> str:
        names = [labels[k] for k, result in driver_analysis.items()
                 if any(f["feature"] == feature and f["q"] < 0.05 for f in result["features"])]
        return join_words(names) if names else "none of the outcomes"

    return {"SEQ_GAP_SURVIVES": survives(FEATURE_LABELS["velocity_gap_per_switch"]),
            "SEQ_REPEAT_SURVIVES": survives(FEATURE_LABELS["repeat_pct"])}


MEMBERSHIP_KEYS = {"+gb": "MEM_GB", "+align": "MEM_ALIGN", "+calledstrike": "MEM_CS", "-whiffmiss": "MEM_NO_WHIFFMISS",
                    "-weak": "MEM_NO_WEAK", "-timing": "MEM_NO_TIMING"}


def membership_text(check: dict) -> dict:
    """Page-text figures from membership_check.json: for each variant composite, its split-half reliability and
    year-over-year correlation, and the change in year-over-year among qualified pairs with its 95% interval."""
    text = {}
    for variant, prefix in MEMBERSHIP_KEYS.items():
        e = check[variant]
        text[f"{prefix}_REL"] = fmt_r(e["reliability"])
        text[f"{prefix}_YOY"] = fmt_r(e["yoy_all"])
        text[f"{prefix}_DIFF"] = f"{e['diff']:+.3f}"
        text[f"{prefix}_CI"] = f"{e['diff_lo']:+.3f} to {e['diff_hi']:+.3f}"
    return text


def projection_text(projection: dict) -> dict:
    """Page-text figures from projection.json: the fitted coefficients, the out-of-sample errors of the projection against
    the raw score and against guessing 100, and what the second season adds."""
    b, c = projection["backtest"], projection["coefficients"]
    one, two = c["one_season"], c["two_seasons"]
    includes_zero = b["two_season_gain_lo"] <= 0 <= b["two_season_gain_hi"]
    return {
        "PROJ_NEXT_SEASON": str(projection["latest_season"] + 1),
        "PROJ_N": f"{b['pairs']:,}",
        "PROJ_TRIPLES": f"{b['triples']:,}",
        "PROJ_RMSE": f"{b['rmse_projection']:.1f}",
        "PROJ_RMSE_RAW": f"{b['rmse_raw_score']:.1f}",
        "PROJ_RMSE_LEAGUE": f"{b['rmse_league_average']:.1f}",
        "PROJ_COVER": f"{b['coverage_80']:.0%}",
        "PROJ_RMSE_ONE": f"{b['triples_rmse_one_season']:.2f}",
        "PROJ_RMSE_TWO": f"{b['triples_rmse_two_seasons']:.2f}",
        "PROJ_TWO_GAIN": f"{b['two_season_gain']:+.2f} points, 95% interval {b['two_season_gain_lo']:+.2f} to {b['two_season_gain_hi']:+.2f}",
        "PROJ_TWO_NOTE": ("that interval includes zero, so the earlier season is the weaker part of the projection" if includes_zero
                          else "that interval excludes zero"),
        "PROJ_COEF_ONE": f"{one['intercept']:.1f} + {one['this']:.2f} × this season",
        "PROJ_COEF_TWO": f"{two['intercept']:.1f} + {two['this']:.2f} × this season + {two['previous']:.2f} × the season before",
    }


def join_labels(labels: list[str], none: str = "no outcome") -> str:
    return join_words(labels) if labels else none


def outcome_text(validation: dict) -> dict:
    """Page-text figures from outcome_validation.json: the sample, which outcomes Deception+ adds to (an interval
    above zero) beyond Stuff+ and Location+ and beyond last season's own result, the stability of each metric, and the
    standing of the timing demotion rule."""
    outcomes = validation["outcomes"]
    first = outcomes["k_pct"]
    clear = lambda key: [o["label"] for o in outcomes.values() if o[key]["lo"] > 0]
    r = validation["stability"]["r"]
    order = ["stuff_plus", "whiff_rate", "deception_plus", "k_pct", "location_plus", "bb_pct", "xwoba", "woba", "rv100"]
    names = {"stuff_plus": "Stuff+", "whiff_rate": "whiff rate", "deception_plus": "Deception+", "k_pct": "strikeout rate", "location_plus": "Location+",
             "bb_pct": "walk rate", "xwoba": "xwOBA allowed", "woba": "wOBA allowed", "rv100": "run value per 100 pitches"}
    stability = ", ".join(f"{names[k]} {r[k]:.2f}" for k in sorted(order, key=lambda k: -r[k]))
    verdict = ("Deception+ is less stable than the raw whiff rate it is built from, which is expected of a residual: what it offers is the part of "
               "results that stuff and location do not explain, not a better forecast of whiffs." if r["deception_plus"] < r["whiff_rate"] else
               "Deception+ is at least as stable as the raw whiff rate it is built from.")
    rule = validation["timing_rule"]
    coef = validation["coefficients"]

    def per_sd(outcome: str, member: str) -> str:
        c = coef[outcome][member]
        scale = 100 if outcome == "k_pct" else 1000
        return f"{c['coef'] * scale:+.1f} points (95% interval {c['lo'] * scale:+.1f} to {c['hi'] * scale:+.1f})"

    if rule["demote"]:
        timing = ("Timing meets the written demotion rule on both strikeout rate and xwOBA: the upper end of its interval is under "
                  f"{rule['share']:.0%} of whiff's effect on each. It should be reviewed for removal from the score.")
    else:
        timing = (f"Timing is the most reliable member, so its effect on the field is worth watching. Per standard deviation it moves the next season's strikeout rate by "
                  f"{per_sd('k_pct', 'timing_index')} and xwOBA allowed (where lower is better) by {per_sd('xwoba', 'timing_index')}, against {per_sd('k_pct', 'whiff_index')} and "
                  f"{per_sd('xwoba', 'whiff_index')} for whiff. The rule written down for timing is to demote it from the score if, on both outcomes, the upper end of its "
                  f"interval is under {rule['share']:.0%} of whiff's effect. That rule is not met "
                  f"(ratios {rule['outcomes']['k_pct']['ratio']:+.2f} for strikeout rate and {rule['outcomes']['xwoba']['ratio']:+.2f} for xwOBA), so it stays in.")
    return {
        "OV_PAIRS": f"{first['n']:,}", "OV_PITCHERS": f"{first['pitchers']:,}", "OV_MIN_PA": str(validation["min_next_pa"]),
        "OV_SUMMARY": (f"Beyond Stuff+ and Location+, Deception+ adds clearly (95% interval above zero) for {join_labels(clear('deception_over_stuff_location'))}. "
                       f"Beyond last season's own result as well, it adds clearly for {join_labels(clear('deception_over_own_stuff_location'))}."),
        "OV_STABILITY": f"Year-over-year correlation among {validation['stability']['n']:,} pairs of pitchers qualified in both seasons: {stability}. {verdict}",
        "OV_TIMING": timing,
    }


def pitch_type_text(summary: dict) -> dict:
    """Page text from pitch_type_scores.json: the display minimum and the range and extremes of the split-half
    reliability of the per-pitch-type whiff and chase scores."""
    from pitch_type_scores import PITCH_TYPE_NAMES, SHOW_MIN_N

    text = {"PT_MIN_N": str(SHOW_MIN_N)}
    for label in ("whiff", "chase"):
        rel = {t: v[label]["reliability"] for t, v in summary["types"].items() if v[label]["reliability"] is not None}
        best, worst = max(rel, key=rel.get), min(rel, key=rel.get)
        key = label.upper()
        text[f"PT_{key}_REL_RANGE"] = f"{rel[worst]:.2f} to {rel[best]:.2f}"
        text[f"PT_{key}_BEST"] = f"{PITCH_TYPE_NAMES[best]} ({rel[best]:.2f})"
        text[f"PT_{key}_WORST"] = f"{PITCH_TYPE_NAMES[worst]} ({rel[worst]:.2f})"
    text["PT_MEDIAN_N"] = f"{pd.Series([v['whiff']['median_n'] for v in summary['types'].values()]).median():.0f}"
    return text


COVERAGE_SD_TOLERANCE = 0.08     # the page says the intervals held only if every z SD (overall and by third) is within this of 1 ...
COVERAGE_95_MINIMUM = 0.93       # ... and at least this share of errors fall within 1.96


def coverage_text(coverage: dict) -> dict:
    """Page text from interval_coverage.json: how far the out-of-sample standardized errors are from SD 1 and 95% coverage, over
    whiff, chase and weak contact, and the largest SD in any third of the sample size or of the scored pitches per game."""
    sds = [r["z_sd"] for r in coverage.values()]
    thirds = [sd for r in coverage.values() for sd in r["z_sd_by_sample_third"] + r.get("z_sd_by_cluster_third", [])]
    within = [r["within_95"] for r in coverage.values()]
    held = max(abs(sd - 1) for sd in sds + thirds) <= COVERAGE_SD_TOLERANCE and min(within) >= COVERAGE_95_MINIMUM
    return {"COV_VERDICT": "They held" if held else "They did not all hold",
            "COV_SD_RANGE": f"{min(sds):.2f} to {max(sds):.2f}", "COV_SD_THIRD_MAX": f"{max(thirds):.2f}",
            "COV_95_RANGE": f"{min(within):.1%} to {max(within):.1%}".replace(".0%", "%"),
            "COV_N": f"{min(r['n'] for r in coverage.values()):,}"}


def location_text(ps: pd.DataFrame, validation: dict) -> dict:
    """Page text on Location+: how it correlates with Deception+, Stuff+ and the members among qualified pitcher-seasons
    (negatively, unlike the other scores), and what it does on the field in the member regression."""
    q = ps[ps["qualified"] & ps["location_plus"].notna()]
    corr = lambda col: fmt_r(float(q["location_plus"].corr(q[col])))
    coef = validation["coefficients"]
    k, x = coef["k_pct"]["location_plus"], coef["xwoba"]["location_plus"]
    return {
        "LOC_DECEPTION_R": corr("deception_plus"), "LOC_STUFF_R": corr("stuff_plus"), "LOC_WHIFFMISS_R": corr("whiffmiss_index"),
        "LOC_WHIFF_R": corr("whiff_index"), "LOC_CHASE_R": corr("chase_index"),
        "LOC_XWOBA": f"{x['coef'] * 1000:+.1f} points of xwOBA allowed (lower is better; 95% interval {x['lo'] * 1000:+.1f} to {x['hi'] * 1000:+.1f}, p {fmt_p(x['p'])})",
        "LOC_K_P": fmt_p(k["p"]),
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
    """Correlation of a component's index across consecutive seasons, pooled over every pair, with both seasons
    held to the same qualifying minimum. (A partial season would need a lower bar on its side, and pairing a
    fully qualified season with a noisier one measurably diluted these correlations, +0.004 to +0.010 once
    matched.)"""
    eligible = ps[ps[f"{key}_n"].fillna(0) >= QUALIFY_MIN_N[key]]
    return pooled_correlation(consecutive_pairs(eligible, [f"{key}_index"]), f"{key}_index")


def check_page_text(text: dict, components: list) -> None:
    """Refuse to publish a figure that came out as nan, inf or None (an empty sample, a merge that found nothing). Every
    placeholder value on the pages and every cell of the validation table is checked."""
    non_finite = re.compile(r"(?i)\b(nan|inf|none)\b")
    bad = {k: v for k, v in text.items() if non_finite.search(str(v))}
    for c in components:
        bad.update({f"{c['key']}.{k}": v for k, v in c.items() if isinstance(v, str) and non_finite.search(v)})
    if bad:
        raise ValueError(f"page text has non-finite figures, refusing to write site_stats.json: {bad}")


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
        "raw/statcast_pitch_level_2024_2026.csv",
        usecols=["pitcher", "season", "game_pk", "game_date", "game_type", "p_throws"],
        low_memory=False,
    )
    raw = raw[raw["game_type"] == "R"]                      # roles and the cutoff follow the scored regular season
    data_through = pd.to_datetime(raw["game_date"]).max()
    hands = raw.drop_duplicates("pitcher").set_index("pitcher")["p_throws"]

    ps = ps.merge(pitcher_roles(raw, usage=ps[["pitcher", "season", "games", "games_started"]]),
                  on=["pitcher", "season"], how="left")
    qualified = ps[ps["qualified"]].copy()

    # year over year, same definition as reliability_and_ci.py
    scored = ps.dropna(subset=["deception_plus"])
    yoy = consecutive_pairs(scored, ["deception_plus"])
    yoy_r = pooled_correlation(yoy, "deception_plus")

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
    with open("output/design_effects.json", encoding="utf-8") as f:
        curves = json.load(f)
    design_effects = [factor for k, _ in COMPONENTS for _, factor in curves[k]]
    label_of = dict(COMPONENTS)

    stuff = stuff_plus_correlations(ps)
    sequencing = sequencing_feature_overlap(pd.read_csv("output/driver_features.csv"), ps)
    clear = clear_of_average_counts(ps)
    yoy_q = year_over_year_summary(ps)
    stuff_lo, stuff_hi = stuff_plus_correlation_interval(ps)
    hand = handedness_gap(ps, hands)

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
            "p": fmt_p(float(pv.loc[key, "p_value"])),
            "n": int(pv.loc[key, "n"]),
            "passes": bool(pv.loc[key, "p_value"] < 0.01),
            "in_score": key in COMPOSITE_OUTCOMES,
        })
    components.append({
        "key": "composite",
        "label": "Deception+ (forecast is for next-season whiff rate)",
        "reliability": fmt_r(float(rr.loc["deception_plus", "r_full_spearman_brown"])),
        "yoy": fmt_r(yoy_r),
        "in_score": True,
        "delta": fmt_delta(float(pv.loc["composite_to_whiff", "cv_delta_r2"])),
        "p": fmt_p(float(pv.loc["composite_to_whiff", "p_value"])),
        "n": int(pv.loc["composite_to_whiff", "n"]),
        "passes": bool(pv.loc["composite_to_whiff", "p_value"] < 0.01),
    })

    with open("output/artifact_data.json", encoding="utf-8") as f:
        n_points = len(json.load(f)["points"])
    with open("output/driver_analysis.json", encoding="utf-8") as f:
        driver_analysis = json.load(f)
    driver_r2 = [m["r2_mean"] for m in driver_analysis.values()]
    with open("output/sequencing_driver_report.json", encoding="utf-8") as f:
        seq_r2 = [m["r2"] for m in json.load(f).values()]
    with open("output/pitch_type_scores.json", encoding="utf-8") as f:
        pitch_type_summary = json.load(f)
    with open("output/interval_coverage.json", encoding="utf-8") as f:
        interval_coverage = json.load(f)
    with open("output/model_validation.json", encoding="utf-8") as f:
        validation = json.load(f)
    with open("output/membership_check.json", encoding="utf-8") as f:
        membership = json.load(f)
    with open("output/projection.json", encoding="utf-8") as f:
        projection = json.load(f)
    with open("output/outcome_validation.json", encoding="utf-8") as f:
        outcome_validation = json.load(f)

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
        "P_CALLEDSTRIKE": fmt_p(float(pv.loc["calledstrike", "p_value"])),
        "YOY_ALIGN": fmt_r(yoy_by_component["align"]),
        "YOY_CALLEDSTRIKE": fmt_r(yoy_by_component["calledstrike"]),
        "P_ALIGN": fmt_p(float(pv.loc["align", "p_value"])),
        "DELTA_ALIGN": fmt_delta(float(pv.loc["align", "cv_delta_r2"])),
        "N_COMPOSITE": str(len(COMPOSITE_OUTCOMES)),
        "QUALIFY_TEXT": qualify_text(),
        "TIMING_ONTIME_R": fmt_r(timing_vs_savant(ps)),
        "DRIVER_R2_RANGE": f"{min(driver_r2):.2f} to {max(driver_r2):.2f}",
        "SEQ_R2_RANGE": f"{min(seq_r2):.4f} to {max(seq_r2):.4f}",
        "STUFF_R": fmt_r(stuff["composite"]),
        "STUFF_R_BY_SEASON": join_words([f"{fmt_r(r)} in {season}" for season, r in sorted(stuff["by_season"].items())]),
        "YOY_BY_PAIR": join_words([f"{fmt_r(r)} for {pair.replace('-', ' to ')}" for pair, r in correlation_by_pair(yoy, "deception_plus").items()]),
        "YOY_QUAL_BY_PAIR": join_words([f"{fmt_r(r)} for {pair.replace('-', ' to ')}" for pair, r in yoy_q["by_pair"].items()]),
        "STUFF_N": f"{stuff['n']:,}",
        "STUFF_R_MEMBERS": join_words([f"{QUALIFY_LABELS[k]} {fmt_r_short(r)}" for k, r in stuff["members"].items()]),
        "DELTA_CALLEDSTRIKE": fmt_delta(float(pv.loc["calledstrike", "cv_delta_r2"])),
        "UNSCORED_PCT": f"{unscored_pitch_share() * 100:.1f}%",
        "GB_FORECAST_R": fmt_r(ground_ball_forecast_correlation(ps)),
        "DP_HALF": f"{((qualified['deception_plus_ci_hi'] - qualified['deception_plus_ci_lo']) / 2).median():.0f}",
        "YOY_QUAL_R": fmt_r(yoy_q["r"]),
        "YOY_QUAL_N": f"{yoy_q['n']:,}",
        "YOY_QUAL_LO": f"{yoy_q['lo']:.2f}",
        "YOY_QUAL_HI": f"{yoy_q['hi']:.2f}",
        "YOY_HIGH_VOL_R": f"{yoy_q['high_volume_r']:.2f}",
        "YOY_LOW_VOL_R": f"{yoy_q['low_volume_r']:.2f}",
        "STUFF_R_CI": f"{stuff_lo:.2f} to {stuff_hi:.2f}",
        "LHP_GAP": f"{hand['gap']:+.1f}",
        "LHP_P": fmt_p(hand["p"]),
        "LHP_N": f"{hand['n_left']:,}",
        "RHP_N": f"{hand['n_right']:,}",
        "P_WEAK": fmt_p(float(pv.loc["weak", "p_value"])),
        "DP_CLEAR": f"{clear[0]:,}",
        "DP_CLEAR_PCT": f"{clear[0] / clear[1]:.0%}",
        "SEQ_FEATURES_R": fmt_r(sequencing["between"]),
        "SEQ_GAP_WHIFF_R": fmt_r(sequencing["gap_whiff"]),
        "SEQ_REPEAT_WHIFF_R": fmt_r(sequencing["repeat_whiff"]),
        "SEQ_SPREAD_R": fmt_r(speed_spread_correlation(pd.read_csv("output/driver_features.csv"), pd.read_csv("output/pitcher_pitchtype_season.csv"))),
        **sequencing_survivors(driver_analysis),
        **validation_text(validation),
        **membership_text(membership),
        **projection_text(projection),
        **outcome_text(outcome_validation),
        **location_text(ps, outcome_validation),
        **coverage_text(interval_coverage),
        **pitch_type_text(pitch_type_summary),
    }

    check_page_text(text, components)
    out = {"text": text, "components": components}
    with open("output/site_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"Saved site_stats.json ({len(text)} text fields, {len(components)} table rows).")
    print(f"data through {text['DATA_THROUGH']}; reliever mean {text['RP_MEAN']} vs starter mean {text['SP_MEAN']}; "
          f"{text['RP_TOP25']} of top 25 qualified are relievers.")


if __name__ == "__main__":
    main()
