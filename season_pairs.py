"""
Pitcher Deception Project: consecutive-season pairs
===================================================
Year-over-year stability and the forecast test both pair a pitcher's season with their next one. With three
seasons that is two pairs (2024 to 2025 and 2025 to 2026), and a pitcher who threw all three contributes to
both. These helpers build the pairs once, so every statistic pools them the same way, and resample whole
pitchers when they need an interval: a pitcher's two pairs share a season and are not independent.
"""

import numpy as np
import pandas as pd


def consecutive_pairs(ps: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per pitcher and pair of consecutive seasons, with `<column>_1` from the earlier season and
    `<column>_2` from the next one, plus `season` (the earlier). Rows missing any column are dropped, so pass
    only the pitcher-seasons that should count (already filtered to qualified, say)."""
    first = ps[["pitcher", "season", *columns]]
    if first.duplicated(["pitcher", "season"]).any():
        raise ValueError("consecutive_pairs needs one row per pitcher and season; a repeated pair would multiply its pairs")
    second = first.assign(season=first["season"] - 1)
    return first.merge(second, on=["pitcher", "season"], suffixes=("_1", "_2")).dropna().reset_index(drop=True)


def pooled_correlation(pairs: pd.DataFrame, column: str) -> float:
    """Correlation between a column's earlier and next-season values over every pair."""
    return float(pairs[f"{column}_1"].corr(pairs[f"{column}_2"]))


def correlation_by_pair(pairs: pd.DataFrame, column: str) -> dict[str, float]:
    """The same correlation for each pair of seasons separately, keyed like "2024-2025"."""
    return {f"{season}-{season + 1}": pooled_correlation(group, column) for season, group in pairs.groupby("season")}


def cluster_bootstrap_correlation_difference(x1: np.ndarray, y1: np.ndarray, x0: np.ndarray, y0: np.ndarray,
                                             pitchers: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """corr(x1, y1) - corr(x0, y0) with its 95% interval, resampling the same whole pitchers for both, so the
    interval is for the difference between two scores computed on the same pairs (mean, low, high)."""
    order = np.argsort(pitchers, kind="stable")
    x1, y1, x0, y0, pitchers = x1[order], y1[order], x0[order], y0[order], pitchers[order]
    _, starts = np.unique(pitchers, return_index=True)
    spans = [np.arange(a, b) for a, b in zip(starts, list(starts[1:]) + [len(pitchers)])]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([spans[i] for i in rng.integers(0, len(spans), len(spans))])
        draws.append(np.corrcoef(x1[idx], y1[idx])[0, 1] - np.corrcoef(x0[idx], y0[idx])[0, 1])
    return float(np.mean(draws)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cluster_bootstrap_interval(x: np.ndarray, y: np.ndarray, pitchers: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float]:
    """95% interval for corr(x, y) from resampling whole pitchers, each with every row they have."""
    order = np.argsort(pitchers, kind="stable")
    x, y, pitchers = x[order], y[order], pitchers[order]
    _, starts = np.unique(pitchers, return_index=True)
    spans = [np.arange(a, b) for a, b in zip(starts, list(starts[1:]) + [len(pitchers)])]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([spans[i] for i in rng.integers(0, len(spans), len(spans))])
        draws.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
