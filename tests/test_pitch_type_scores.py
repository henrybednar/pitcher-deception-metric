import numpy as np
import pandas as pd
import pytest

import pitch_type_scores as pts


def swings(n_pitchers: int = 90) -> pd.DataFrame:
    """Four-seamers and sliders for many pitchers, each with a little talent above or below the expectation. Pitcher 1's
    sliders whiff far above it."""
    rng = np.random.default_rng(11)
    rows = []
    for pitcher in range(1, n_pitchers + 1):
        for pitch_type, expected in (("FF", 0.22), ("SL", 0.35)):
            actual = expected + (0.2 if (pitcher, pitch_type) == (1, "SL") else rng.normal(0, 0.04))
            for i in range(240):
                game = 1000 + pitcher * 100 + i % 24
                rows.append(dict(pitcher=pitcher, season=2025, game_pk=game, half=game % 2, pitch_type=pitch_type, is_swing=True,
                                 is_in_zone=False, is_bip=False, is_whiff=int(rng.random() < actual), whiff_expected_full=expected))
    return pd.DataFrame(rows)


def test_a_pitch_type_is_scored_against_the_same_type_only():
    df = swings()

    scored, summary = pts.score_pitch_type(df, "SL", "whiff")

    top = scored.sort_values("whiff_index").iloc[-1]
    assert top["pitcher"] == 1 and top["pitch_type"] == "SL"           # the pitcher whose slider beats its expectation
    assert top["whiff_n"] == 240                                        # sliders only, not the pitcher's 480 swings
    assert top["whiff_index"] > 110
    assert summary["n_pitcher_seasons"] == len(scored) == 90
    assert summary["median_n"] == 240


def test_score_all_returns_one_row_per_pitcher_season_and_pitch_type_with_each_labels_columns(monkeypatch):
    monkeypatch.setattr(pts, "PITCH_TYPES", ("FF", "SL"))
    monkeypatch.setattr(pts, "LABELS", ("whiff",))

    table, summary = pts.score_all(swings())

    assert len(table) == 180
    assert set(table["pitch_type"]) == {"FF", "SL"}
    assert {"whiff_n", "whiff_index", "whiff_ci_lo", "whiff_ci_hi"} <= set(table.columns)
    assert set(summary) == {"FF", "SL"} and set(summary["SL"]) == {"whiff"}


def table_row(pitch_type, whiff_n, whiff_index, chase_n, chase_index, pitcher=1, season=2025):
    return dict(pitcher=pitcher, season=season, pitch_type=pitch_type, whiff_n=whiff_n, whiff_index=whiff_index,
                chase_n=chase_n, chase_index=chase_index)


def test_leaderboard_lists_hide_thin_indexes_drop_empty_types_and_put_the_most_used_first():
    table = pd.DataFrame([
        table_row("SL", 120, 108.26, 80, 99.04),
        table_row("FF", 300, 101.0, 280, 97.5),
        table_row("CU", 40, 120.0, 30, 130.0),                        # under the display minimum on both: left out
        table_row("CH", 70, 104.0, 20, 140.0),                        # whiff shown, chase too thin to show
        table_row("FS", 60, np.nan, np.nan, np.nan),                   # no scores at all
        table_row("FF", 200, 99.0, 150, 100.0, pitcher=2),
    ])

    out = pts.leaderboard_lists(table)

    assert [e[0] for e in out[(1, 2025)]] == ["FF", "SL", "CH"]
    assert out[(1, 2025)][1] == ["SL", 120, 108.3, 80, 99.0]
    assert out[(1, 2025)][2] == ["CH", 70, 104.0, 20, None]            # the sample size stays even when the index is hidden
    assert list(out[(2, 2025)]) == [["FF", 200, 99.0, 150, 100.0]]
    assert (3, 2025) not in out


def test_leaderboard_lists_keep_a_type_with_a_chase_score_and_no_whiff_sample():
    table = pd.DataFrame([table_row("SI", np.nan, np.nan, 90, 103.0)])

    out = pts.leaderboard_lists(table)

    assert out[(1, 2025)] == [["SI", None, None, 90, 103.0]]
