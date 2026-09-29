import numpy as np
import pandas as pd
import pytest

from driver_features import circular_mirror_score
from site_common import fill_stats


def test_mirror_score_is_zero_for_match_and_for_mirror_and_90_for_perpendicular():
    scores = circular_mirror_score(np.array([180.0, 180.0, 0.0]), np.array([180.0, 0.0, 90.0]))

    assert scores.tolist() == [0.0, 0.0, 90.0]


def test_mirror_score_is_nan_when_either_axis_is_missing():
    scores = circular_mirror_score(np.array([10.0, np.nan]), np.array([np.nan, 20.0]))

    assert np.isnan(scores).all()


def test_fill_stats_replaces_known_keys_and_escapes_html():
    assert fill_stats("r={{R}} <{{NAME}}>", {"R": "0.617", "NAME": "A&B"}) == "r=0.617 <A&amp;B>"


def test_fill_stats_fails_on_a_key_with_no_value():
    with pytest.raises(KeyError, match="MISSING_KEY"):
        fill_stats("{{MISSING_KEY}}", {})
