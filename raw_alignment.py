"""
Pitcher Deception Project: raw pitch columns lined up with the scored pitches
=============================================================================
per_pitch_predictions.csv holds the regular-season rows of the raw pitch file in the raw file's own
order, with only the columns the fit steps need. A diagnostic that wants another raw column (the pitch
number in the plate appearance, the count, the previous pitch) reads it from the raw file and attaches it
by position. regular_season_rows() does that and refuses to go on unless pitcher and season agree on
every row, the same guard read_aligned_predictions() applies when the fit steps append columns.
"""

import pandas as pd


def regular_season_rows(raw_path: str, columns: list[str], scored: pd.DataFrame) -> pd.DataFrame:
    """The requested raw columns for regular-season pitches, indexed like `scored`.

    Raises ValueError when the row count differs from `scored` or any row's pitcher or season does not
    match, because a position-based merge on rows that do not line up would silently attach the wrong values."""
    raw = pd.read_csv(raw_path, usecols=["pitcher", "season", "game_type"] + list(columns), low_memory=False)
    raw = raw[raw["game_type"] == "R"].reset_index(drop=True)
    if (len(raw) != len(scored) or not (raw["pitcher"].to_numpy() == scored["pitcher"].to_numpy()).all()
            or not (raw["season"].to_numpy() == scored["season"].to_numpy()).all()):
        raise ValueError("raw pitch rows do not line up with the scored pitches; refusing to attach columns by position")
    out = raw[list(columns)].copy()
    out.index = scored.index
    return out
