"""
Pitcher Deception Project: does the composite's membership still hold?
======================================================================
Deception+ has five members. This step rescores the composite with one member dropped or one candidate added
(ground ball, horizontal alignment, called strike) and compares the result with the shipped composite on the same
pitcher-seasons: split-half reliability, year-over-year correlation over every scored pair of consecutive seasons,
and year-over-year correlation among pairs where the pitcher qualified in both seasons, with a 95% interval for
the difference from the shipped composite that resamples whole pitchers. The weights are the same as the
composite's (each member's league-wide reliability times the pitcher's sample over that member's median).

Run after reliability_and_ci.py, which writes the half-season scores this reads. Writes
output/membership_check.json; export_site_stats.py turns it into page text, so the figures quoted next to the
membership decisions cannot go stale.
"""

import json

import numpy as np
import pandas as pd

from reliability_and_ci import COMPOSITE_OUTCOMES, compute_composite
from season_pairs import cluster_bootstrap_correlation_difference, consecutive_pairs, correlation_by_pair, pooled_correlation

N_BOOT = 600
CANDIDATES = {"gb": "ground ball", "align": "alignment", "calledstrike": "called strike"}


def variant_labels() -> dict[str, list[str]]:
    """The shipped composite, each member dropped in turn, and each candidate added alone."""
    members = list(COMPOSITE_OUTCOMES)
    variants = {"shipped": members}
    variants.update({f"-{m}": [x for x in members if x != m] for m in members})
    variants.update({f"+{c}": members + [c] for c in CANDIDATES})
    return variants


def composite_score(ps: pd.DataFrame, labels: list[str], weights: dict[str, float]) -> np.ndarray:
    """Deception+ for `labels` on the scale of the qualified pitcher-seasons (mean 100, SD 10), NaN where fewer
    than two of the labels are scored."""
    raw, _, eligible = compute_composite(ps, [f"{l}_index" for l in labels], [f"{l}_n" for l in labels], [weights[l] for l in labels])
    pop = pd.Series(raw).loc[eligible & ps["qualified"].to_numpy()]
    return np.where(eligible, 100 + 10 * (raw - pop.mean()) / pop.std(), np.nan)


def split_half_reliability(half: pd.DataFrame, qualified: np.ndarray, labels: list[str], weights: dict[str, float]) -> float:
    """Spearman-Brown corrected correlation between the composite from odd games and from even games."""
    scores = []
    for h in (0, 1):
        raw, _, eligible = compute_composite(half, [f"{l}_index_h{h}" for l in labels], [f"{l}_n_h{h}" for l in labels],
                                             [weights[l] for l in labels])
        sd = pd.Series(raw).loc[eligible & qualified].std()
        scores.append(np.where(eligible, 100 + 10 * raw / sd, np.nan))
    ok = ~np.isnan(scores[0]) & ~np.isnan(scores[1])
    r = np.corrcoef(scores[0][ok], scores[1][ok])[0, 1]
    return float(2 * r / (1 + r))


def compare_variants(ps: pd.DataFrame, half: pd.DataFrame, weights: dict[str, float], n_boot: int = N_BOOT) -> dict:
    lookup = ps.set_index(["pitcher", "season"])["qualified"]
    half_qualified = half.set_index(["pitcher", "season"]).index.map(lookup).fillna(False).to_numpy(dtype=bool)
    out, base = {}, None
    for name, labels in variant_labels().items():
        scored = ps.assign(dp=composite_score(ps, labels, weights)).dropna(subset=["dp"])[["pitcher", "season", "dp", "qualified"]]
        all_pairs = consecutive_pairs(scored[["pitcher", "season", "dp"]], ["dp"])
        qualified_pairs = consecutive_pairs(scored[scored["qualified"]][["pitcher", "season", "dp"]], ["dp"])
        entry = {"labels": labels, "reliability": split_half_reliability(half, half_qualified, labels, weights),
                 "yoy_all": pooled_correlation(all_pairs, "dp"), "yoy_by_pair": correlation_by_pair(all_pairs, "dp"),
                 "yoy_qualified": pooled_correlation(qualified_pairs, "dp"), "n_pairs": int(len(qualified_pairs))}
        if base is None:
            base = qualified_pairs
        else:
            both = qualified_pairs.merge(base, on=["pitcher", "season"], suffixes=("", "_base"))
            mean, lo, hi = cluster_bootstrap_correlation_difference(
                both["dp_1"].to_numpy(), both["dp_2"].to_numpy(), both["dp_1_base"].to_numpy(), both["dp_2_base"].to_numpy(),
                both["pitcher"].to_numpy(), n_boot)
            entry.update({"diff": mean, "diff_lo": lo, "diff_hi": hi})
        out[name] = entry
    return out


if __name__ == "__main__":
    ps = pd.read_csv("output/pitcher_season.csv")
    half = pd.read_csv("output/half_scores.csv")
    report = pd.read_csv("output/reliability_report.csv").set_index("label")
    weights = {label: float(report.loc[label, "r_full_spearman_brown"]) for label in list(COMPOSITE_OUTCOMES) + list(CANDIDATES)}

    result = compare_variants(ps, half, weights)
    for name, e in result.items():
        diff = f"  vs shipped {e['diff']:+.3f} [{e['diff_lo']:+.3f}, {e['diff_hi']:+.3f}]" if "diff" in e else ""
        print(f"{name:16s} reliability {e['reliability']:.3f}  year-over-year {e['yoy_all']:.3f}  "
              f"qualified pairs {e['yoy_qualified']:.3f} (n={e['n_pairs']}){diff}", flush=True)
    with open("output/membership_check.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Saved membership_check.json. Done.")
