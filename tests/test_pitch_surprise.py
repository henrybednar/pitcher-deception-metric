import numpy as np
import pandas as pd
import pytest

import pitch_surprise as psu


def raw_rows():
    # one plate appearance thrown out of order: FF (ball), SL (called strike), then a sweeper listed under its old code SV
    return pd.DataFrame({
        "game_pk": 1, "at_bat_number": 5, "pitcher": 9,
        "pitch_number": [3, 1, 2], "pitch_type": ["SV", "FF", "SL"],
        "description": ["swinging_strike", "ball", "called_strike"],
    })


def test_previous_pitch_context_looks_back_in_pitch_order_within_the_plate_appearance():
    out = psu.previous_pitch_context(raw_rows())

    first, second, third = out.iloc[1], out.iloc[2], out.iloc[0]          # by pitch_number 1, 2, 3
    assert (first["prev1"], first["prev2"], first["prev_result"]) == ("NONE", "NONE", "NONE")
    assert (second["prev1"], second["prev2"], second["prev_result"]) == ("FF", "NONE", "ball")
    assert (third["prev1"], third["prev2"], third["prev_result"]) == ("SL", "FF", "called")


def test_previous_pitch_context_never_crosses_plate_appearances_or_pitchers():
    raw = pd.concat([raw_rows(), raw_rows().assign(at_bat_number=6, pitch_number=[1, 2, 3])], ignore_index=True)
    raw.loc[3:, "pitcher"] = 10                                              # a different pitcher in the next plate appearance

    out = psu.previous_pitch_context(raw)

    assert out.iloc[3:]["prev1"].tolist().count("NONE") == 1                  # only its own first pitch starts fresh
    assert "SV" not in set(out["prev1"]) | set(out["prev2"])                  # slurves read as sweepers


def two_strike_pitcher(n=600, seed=0, informed=True):
    rng = np.random.default_rng(seed)
    two_strikes = rng.integers(0, 2, n)
    X = np.column_stack([two_strikes, 1 - two_strikes, rng.integers(0, 2, n)]).astype(float)
    if informed:
        types = np.where(two_strikes == 1, "SL", "FF")
        flip = rng.random(n) < 0.05
        types = np.where(flip, np.where(types == "SL", "FF", "SL"), types)     # 5% of the time he does the unexpected thing
    else:
        types = rng.choice(["FF", "SL"], n)
    return types, X, np.repeat(np.arange(n // 20), 20)


def test_a_pitcher_who_follows_the_count_is_predictable_and_a_random_one_is_not():
    informed = psu.cross_fitted_surprisal(*two_strike_pitcher(informed=True))
    random = psu.cross_fitted_surprisal(*two_strike_pitcher(informed=False))

    assert informed[1].mean() - informed[0].mean() > 0.3                    # context removes most of the uncertainty
    assert abs(random[1].mean() - random[0].mean()) < 0.05                  # nothing to learn, so no gain
    assert informed[0].mean() < random[0].mean()


def test_the_rare_deviation_is_more_surprising_than_the_expected_pitch():
    types, X, games = two_strike_pitcher(seed=1)
    conditional, _ = psu.cross_fitted_surprisal(types, X, games)
    expected = np.where(X[:, 0] == 1, "SL", "FF")

    assert conditional[types != expected].mean() > conditional[types == expected].mean() + 1.5


def test_surprisal_is_nan_for_a_pitcher_with_one_pitch_type_or_too_few_games():
    types, X, games = two_strike_pitcher()
    one_type = psu.cross_fitted_surprisal(np.repeat("FF", len(types)), X, games)
    one_game = psu.cross_fitted_surprisal(types, X, np.zeros(len(types), int))

    assert np.isnan(one_type[0]).all() and np.isnan(one_game[0]).all()


def test_a_pitch_type_the_training_games_never_saw_gets_a_floored_finite_surprisal():
    types = np.array(["FF"] * 100 + ["SL"] * 100 + ["CH"] * 3)              # the changeup shows up in a single game
    games = np.concatenate([np.repeat(np.arange(10), 20), [10, 10, 10]])
    X = np.random.default_rng(0).integers(0, 2, (203, 3)).astype(float)

    conditional, marginal = psu.cross_fitted_surprisal(types, X, games)

    assert np.isfinite(conditional).all() and np.isfinite(marginal).all()
    assert conditional.max() <= -np.log(psu.PROB_FLOOR) + 1e-9


def test_within_cell_slope_recovers_the_effect_even_when_cells_differ_in_both_variables():
    rng = np.random.default_rng(3)
    rows = []
    for cell in range(40):
        shift = cell * 0.5                                                  # cells with more surprise also have higher residuals
        x = rng.normal(shift, 1.0, 80)
        rows.append(pd.DataFrame({"cell": cell, "pitcher": cell // 2, "x": x, "y": 3 * shift + 0.3 * (x - shift) + rng.normal(0, 1, 80)}))
    frame = pd.concat(rows, ignore_index=True)

    result = psu.within_cell_slope(frame, "y", "x", ["cell"], "pitcher")

    assert result["beta"] == pytest.approx(0.3, abs=0.06)
    assert result["lo"] < 0.3 < result["hi"]
    assert result["cells"] == 40 and result["pitchers"] == 20
    assert result["per_sd"] == pytest.approx(result["beta"] * result["sd_x"])


def test_within_cell_slope_drops_cells_that_are_too_small_to_compare_within():
    frame = pd.DataFrame({"cell": [1] * 30 + [2] * 3, "pitcher": 1, "x": np.arange(33.0), "y": np.arange(33.0)})

    assert psu.within_cell_slope(frame, "y", "x", ["cell"], "pitcher", min_rows=10)["cells"] == 1


def test_season_regression_gives_standardized_effects_in_outcome_units():
    rng = np.random.default_rng(4)
    n = 600
    table = pd.DataFrame({"pitcher": np.arange(n) // 2, "marg": rng.normal(1.0, 0.3, n), "gain": rng.normal(0.1, 0.05, n)})
    table["index"] = 100 + 4 * (table["gain"] - 0.1) / 0.05 + rng.normal(0, 8, n)

    result = psu.season_regression(table, "index", ["marg", "gain"])

    assert result["gain"]["per_sd"] == pytest.approx(4.0, abs=1.5)
    assert abs(result["marg"]["per_sd"]) < 1.5
    assert result["gain"]["lo"] < result["gain"]["per_sd"] < result["gain"]["hi"]


def test_regular_season_rows_refuses_rows_that_do_not_line_up(tmp_path):
    from raw_alignment import regular_season_rows

    raw = pd.DataFrame({"pitcher": [1, 2], "season": 2025, "game_type": ["R", "R"], "balls": [0, 1]})
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)

    assert regular_season_rows(str(path), ["balls"], pd.DataFrame({"pitcher": [1, 2], "season": 2025}))["balls"].tolist() == [0, 1]
    with pytest.raises(ValueError):
        regular_season_rows(str(path), ["balls"], pd.DataFrame({"pitcher": [2, 1], "season": 2025}))


def test_role_dummies_use_the_games_started_share_and_leave_unknown_roles_missing():
    ps = pd.DataFrame({"games": [30, 60, 40, 0, np.nan], "games_started": [30, 0, 20, 0, np.nan]})

    out = psu.role_dummies(ps)

    assert out["is_rp"].tolist()[:3] == [0.0, 1.0, 0.0]
    assert out["is_mr"].tolist()[:3] == [0.0, 0.0, 1.0]                 # a starter is neither
    assert out.iloc[3:].isna().all().all()
