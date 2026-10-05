import numpy as np
import pandas as pd
import pytest

import membership_check as mc
from reliability_and_ci import COMPOSITE_OUTCOMES

LABELS = list(COMPOSITE_OUTCOMES) + ["gb", "align", "calledstrike"]


def synthetic(n=400, seed=0):
    """Three seasons of members that share a persistent talent, a ground-ball index that is pure noise each season,
    and half-season indexes that are the same talent with more noise."""
    rng = np.random.default_rng(seed)
    talent = rng.normal(0, 1, n)
    rows, halves = [], []
    for season in (2024, 2025, 2026):
        ps = pd.DataFrame({"pitcher": np.arange(n), "season": season, "qualified": True})
        half = pd.DataFrame({"pitcher": np.arange(n), "season": season})
        for label in LABELS:
            signal = 0.0 if label in ("gb", "align", "calledstrike") else 1.0
            ps[f"{label}_index"] = 100 + 10 * (signal * 0.8 * talent + rng.normal(0, 0.8, n))
            ps[f"{label}_n"] = 200.0
            for h in (0, 1):
                half[f"{label}_index_h{h}"] = 100 + 10 * (signal * 0.8 * talent + rng.normal(0, 1.1, n))
                half[f"{label}_n_h{h}"] = 100.0
        rows.append(ps)
        halves.append(half)
    return pd.concat(rows, ignore_index=True), pd.concat(halves, ignore_index=True), {l: 0.5 for l in LABELS}


def test_variants_are_the_shipped_composite_each_member_dropped_and_each_candidate_added():
    variants = mc.variant_labels()

    assert variants["shipped"] == list(COMPOSITE_OUTCOMES)
    assert variants["-whiff"] == [m for m in COMPOSITE_OUTCOMES if m != "whiff"]
    assert variants["+gb"] == list(COMPOSITE_OUTCOMES) + ["gb"]
    assert len(variants) == 1 + len(COMPOSITE_OUTCOMES) + 3


def test_composite_score_is_centred_on_the_qualified_pitcher_seasons_at_100_with_sd_10():
    ps, _, weights = synthetic()
    ps.loc[ps["pitcher"] < 50, "qualified"] = False

    score = mc.composite_score(ps, list(COMPOSITE_OUTCOMES), weights)

    q = ps["qualified"].to_numpy()
    assert np.nanmean(score[q]) == pytest.approx(100, abs=1e-6)
    assert np.nanstd(score[q], ddof=1) == pytest.approx(10, abs=1e-6)


def test_adding_a_member_with_no_persistent_talent_lowers_reliability_and_year_over_year_with_an_interval_below_zero():
    ps, half, weights = synthetic()

    result = mc.compare_variants(ps, half, weights, n_boot=200)

    assert result["+gb"]["reliability"] < result["shipped"]["reliability"]
    assert result["+gb"]["yoy_all"] < result["shipped"]["yoy_all"]
    assert result["+gb"]["diff_hi"] < 0
    assert result["-whiff"]["reliability"] < result["shipped"]["reliability"]       # dropping real signal costs stability too
    assert result["shipped"]["n_pairs"] == 800 and set(result["shipped"]["yoy_by_pair"]) == {"2024-2025", "2025-2026"}


def test_split_half_reliability_applies_the_spearman_brown_correction_to_the_half_to_half_correlation():
    rng = np.random.default_rng(3)
    n = 500
    talent = rng.normal(0, 1, n)
    half = pd.DataFrame({"pitcher": np.arange(n), "season": 2025})
    for label in ("whiff", "chase"):
        for h in (0, 1):
            half[f"{label}_index_h{h}"] = 100 + 10 * (talent + rng.normal(0, 1.5, n))
            half[f"{label}_n_h{h}"] = 100.0

    reliability = mc.split_half_reliability(half, np.ones(n, dtype=bool), ["whiff", "chase"], {"whiff": 0.5, "chase": 0.5})

    r = np.corrcoef(half["whiff_index_h0"] + half["chase_index_h0"], half["whiff_index_h1"] + half["chase_index_h1"])[0, 1]
    assert reliability == pytest.approx(2 * r / (1 + r), abs=1e-9)
    assert reliability > r
