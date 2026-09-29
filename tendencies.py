"""
Pitcher Deception Project: fold-honest batter and catcher tendencies
====================================================================
A hitter's (or catcher's) tendency for an outcome is the mean of that outcome over the other
pitchers they saw that season. It tells the full-tier model who was at bat and who was catching.

Two leaks have to be closed, and both are closed here by building the feature inside each
cross-validation fold:

1. Own-pitcher leak. A catcher sees a median of about 20 pitchers a season, so removing only the
   scored pitch leaves thousands of the same pitcher's pitches in the average, and the feature
   re-encodes that pitcher. Training rows therefore leave out EVERY pitch from their own pitcher.
2. Held-out leak. Building the feature once over all data puts the outcomes of held-out pitchers
   into the features of training rows. Measured on whiff for four-seamers, that made the
   out-of-fold AUC 0.010 too high (0.7581 vs 0.7477 when built fold-honestly). Here the totals
   come only from the fold's training pitchers, and test rows use those totals unchanged.

Usage: build_totals() once per outcome, then tendency_for_rows() inside each fold.
"""

import numpy as np
import pandas as pd

TOTAL_COLS = ["tsum", "tcount"]


def build_totals(df: pd.DataFrame, mask: pd.Series, key_col: str, target_col: str) -> pd.DataFrame:
    """Per (key, season, pitcher) sum and count of the target over the outcome's rows."""
    sub = df.loc[mask, [key_col, "season", "pitcher", target_col]]
    return (sub.groupby([key_col, "season", "pitcher"])[target_col]
            .agg(tsum="sum", tcount="count").reset_index())


def tendency_for_rows(rows: pd.DataFrame, totals: pd.DataFrame, key_col: str,
                      train_pitchers: np.ndarray, leave_out_own: bool) -> np.ndarray:
    """Mean target for each row's (key, season) over the training pitchers.

    leave_out_own=True is for training rows: their own pitcher's contribution is subtracted, so a
    pitcher never sets the feature that scores that pitcher. Test rows use False, because test
    pitchers are not in train_pitchers and contribute nothing to the totals.
    NaN where no other training pitcher contributed.
    """
    in_train = totals[totals["pitcher"].isin(train_pitchers)]
    by_key = in_train.groupby([key_col, "season"])[TOTAL_COLS].sum().rename(columns={"tsum": "ksum", "tcount": "kcount"}).reset_index()
    keyed = rows[[key_col, "season", "pitcher"]].merge(by_key, on=[key_col, "season"], how="left")
    ksum, kcount = keyed["ksum"].fillna(0.0).to_numpy(), keyed["kcount"].fillna(0.0).to_numpy()
    if leave_out_own:
        own = in_train[[key_col, "season", "pitcher", "tsum", "tcount"]].rename(columns={"tsum": "osum", "tcount": "ocount"})
        keyed = keyed.merge(own, on=[key_col, "season", "pitcher"], how="left")
        ksum = ksum - keyed["osum"].fillna(0.0).to_numpy()
        kcount = kcount - keyed["ocount"].fillna(0.0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(kcount > 0, ksum / kcount, np.nan)
