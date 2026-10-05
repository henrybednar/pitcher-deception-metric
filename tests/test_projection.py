import numpy as np
import pandas as pd
import pytest

import projection as pj


def three_seasons(n=400, seed=0):
    """Pitchers with a persistent talent and season-to-season noise, scored in 2024, 2025 and 2026."""
    rng = np.random.default_rng(seed)
    talent = rng.normal(0, 7, n)
    rows = []
    for season in (2024, 2025, 2026):
        rows.append(pd.DataFrame({"pitcher": np.arange(n), "season": season, "deception_plus": 100 + talent + rng.normal(0, 7, n),
                                  "qualified": True}))
    return pd.concat(rows, ignore_index=True)


def test_season_rows_attach_the_previous_and_next_scored_season_and_leave_gaps_empty():
    ps = pd.DataFrame({"pitcher": [1, 1, 1, 2, 2], "season": [2024, 2025, 2026, 2024, 2026],
                       "deception_plus": [90.0, 100.0, 110.0, 80.0, 120.0], "qualified": True})

    rows = pj.season_rows(ps).sort_values(["pitcher", "season"]).reset_index(drop=True)

    assert rows.loc[rows["season"].eq(2025) & rows["pitcher"].eq(1), ["prev", "cur", "nxt"]].iloc[0].tolist() == [90.0, 100.0, 110.0]
    assert np.isnan(rows.loc[rows["season"].eq(2024) & rows["pitcher"].eq(1), "prev"].iloc[0])
    assert np.isnan(rows.loc[rows["season"].eq(2026) & rows["pitcher"].eq(1), "nxt"].iloc[0])
    assert rows.loc[rows["pitcher"].eq(2), ["prev", "nxt"]].isna().all().all()        # pitcher 2 skipped 2025


def test_cross_fitted_projections_beat_the_raw_score_and_give_a_second_projection_only_with_a_previous_season():
    frame = pj.cross_fitted_projections(pj.season_rows(three_seasons()), seeds=(0, 1), n_splits=5)

    assert frame.loc[frame["prev"].isna(), "proj2"].isna().all()
    assert frame.loc[frame["prev"].notna(), "proj2"].notna().all()
    pairs = frame.dropna(subset=["nxt"])
    assert pj.rmse(pairs["nxt"], pairs["proj1"]) < pj.rmse(pairs["nxt"], pairs["cur"]) - 0.5   # shrinking toward 100 helps a lot
    triples = pairs.dropna(subset=["prev"])
    assert pj.rmse(triples["nxt"], triples["proj2"]) < pj.rmse(triples["nxt"], triples["proj1"])  # a second noisy season helps too


def test_a_pitchers_projection_does_not_depend_on_that_pitchers_own_outcomes():
    rows = pj.season_rows(three_seasons(n=80))
    altered = rows.copy()
    altered.loc[altered["pitcher"].eq(0), "nxt"] += 500.0                      # pitcher 0's own next-season results change

    before = pj.cross_fitted_projections(rows, seeds=(0,), n_splits=5)
    after = pj.cross_fitted_projections(altered, seeds=(0,), n_splits=5)

    own = rows["pitcher"].eq(0)
    assert after.loc[own, "proj1"].to_numpy() == pytest.approx(before.loc[own, "proj1"].to_numpy(), abs=1e-9)
    assert not np.allclose(after.loc[~own, "proj1"], before.loc[~own, "proj1"])       # everyone else's coefficients did move


def test_projection_uses_two_seasons_where_it_can_and_the_range_covers_about_eighty_percent_of_outcomes():
    frame = pj.add_projection(pj.cross_fitted_projections(pj.season_rows(three_seasons(n=800, seed=2)), seeds=(0, 1), n_splits=5))

    assert set(frame.loc[frame["prev"].notna(), "proj_basis"]) == {2} and set(frame.loc[frame["prev"].isna(), "proj_basis"]) == {1}
    assert (frame["proj_hi"] > frame["projection"]).all() and (frame["projection"] > frame["proj_lo"]).all()
    outcome = frame.dropna(subset=["nxt"])
    covered = ((outcome["nxt"] >= outcome["proj_lo"]) & (outcome["nxt"] <= outcome["proj_hi"])).mean()
    assert covered == pytest.approx(0.8, abs=0.04)


def test_backtest_reports_the_projection_against_the_raw_score_and_the_league_average():
    frame = pj.add_projection(pj.cross_fitted_projections(pj.season_rows(three_seasons(n=500, seed=3)), seeds=(0, 1), n_splits=5))

    result = pj.backtest(frame, n_boot=200)

    assert result["rmse_projection"] < result["rmse_raw_score"] and result["rmse_projection"] < result["rmse_league_average"]
    assert result["pairs"] == 1000 and result["triples"] == 500
    assert result["two_season_gain_lo"] < result["two_season_gain"] < result["two_season_gain_hi"]
    assert result["r_projection"] == pytest.approx(result["r_raw_score"], abs=0.05)         # same ranking information, better scale
