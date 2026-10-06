import numpy as np
import pandas as pd
import pytest

import interval_coverage as ic

CURVE = [(10.0, 1.0)]


def pitches(talent_sd: float, noise_scale: float = 1.0, n_pitchers: int = 400, seed: int = 4) -> pd.DataFrame:
    """Whiffs for many pitchers, 300 swings each over 30 games, with a true talent above or below a 0.25 expectation.
    noise_scale above 1 adds game-level shocks the analytic variance does not know about."""
    rng = np.random.default_rng(seed)
    rows = []
    for pitcher in range(n_pitchers):
        talent = rng.normal(0, talent_sd)
        for game in range(30):
            p = np.clip(0.25 + talent + rng.normal(0, 0.04 * (noise_scale - 1)), 0.01, 0.99)
            whiffs = rng.random(10) < p
            for w in whiffs:
                rows.append(dict(pitcher=pitcher, season=2025, game_pk=1000 * pitcher + game, half=(1000 * pitcher + game) % 2,
                                 is_swing=True, is_in_zone=False, is_bip=False, is_whiff=int(w), is_weak=0, whiff_expected_full=0.25))
    return pd.DataFrame(rows)


def run(pp):
    wide = ic.half_tables(pp, "whiff")
    return ic.summarize(ic.predictive_z(wide, CURVE))


def test_intervals_are_calibrated_when_the_variance_model_is_right():
    summary = run(pitches(talent_sd=0.03))

    assert summary["z_sd"] == pytest.approx(1.0, abs=0.08)
    assert 0.92 < summary["within_95"] < 0.98


def test_game_level_shocks_the_variance_model_misses_show_up_as_a_z_spread_above_one():
    summary = run(pitches(talent_sd=0.03, noise_scale=6.0))

    assert summary["z_sd"] > 1.4 and summary["within_95"] < 0.9


def test_half_tables_keep_both_halves_of_each_pitcher_season_and_the_games_in_each():
    wide = ic.half_tables(pitches(talent_sd=0.03, n_pitchers=5), "whiff")

    assert len(wide) == 5
    assert (wide["n_games0"] + wide["n_games1"] == 30).all()
    assert (wide["n0"] + wide["n1"] == 300).all()


def test_coverage_keeps_only_qualified_pitcher_seasons_with_enough_pitches_in_both_halves(monkeypatch):
    monkeypatch.setattr(ic, "LABELS", ("whiff",))
    pp = pitches(talent_sd=0.03, n_pitchers=60)
    qualified = pd.DataFrame({"pitcher": range(60), "season": 2025, "qualified": [i % 2 == 0 for i in range(60)]})

    result = ic.coverage(pp, qualified, {"whiff": CURVE})

    assert result["whiff"]["n"] == 30


def test_pitcher_seasons_with_too_few_pitches_in_a_half_are_left_out_of_coverage(monkeypatch):
    monkeypatch.setattr(ic, "LABELS", ("whiff",))
    pp = pitches(talent_sd=0.03, n_pitchers=40)
    thin = pitches(talent_sd=0.03, n_pitchers=10, seed=9).assign(pitcher=lambda d: d["pitcher"] + 1000)
    thin = pd.concat([thin[thin["half"] == 0].groupby("pitcher").head(8), thin[thin["half"] == 1]])      # 8 swings in one half only
    qualified = pd.DataFrame({"pitcher": list(range(40)) + list(range(1000, 1010)), "season": 2025, "qualified": True})

    result = ic.coverage(pd.concat([pp, thin]), qualified, {"whiff": CURVE})

    assert result["whiff"]["n"] == 40


def test_the_design_effect_scales_both_halves_variances_so_a_larger_one_shrinks_the_standardized_errors():
    wide = ic.half_tables(pitches(talent_sd=0.03), "whiff")

    plain = ic.predictive_z(wide, [(10.0, 1.0)])["z"].std()
    inflated = ic.predictive_z(wide, [(10.0, 4.0)])["z"].std()

    assert plain == pytest.approx(1.0, abs=0.08)
    assert inflated < 0.8                                                     # variances four times larger, errors about two thirds as large in SD


def test_sample_thirds_order_by_the_smaller_halfs_pitch_count():
    z = pd.DataFrame({"n_min": range(9), "z": [-1, 1, 0, -2, 2, 0, -3, 3, 0]})

    assert ic.summarize(z)["z_sd_by_sample_third"] == [1.0, 2.0, 3.0]          # the third with the most pitches carries the widest errors here


def test_predictions_run_in_both_directions_and_centre_on_the_shrinkage_target_not_zero():
    rng = np.random.default_rng(2)
    base = pitches(talent_sd=0.03)
    pp = base.assign(is_whiff=np.where(base["is_whiff"] == 1, 1, (rng.random(len(base)) < 0.04).astype(int)))      # the league runs about 3 points above the expectation
    wide = ic.half_tables(pp, "whiff")

    z = ic.predictive_z(wide, CURVE)

    assert len(z) == 2 * len(wide)
    assert not np.allclose(z["z"].iloc[: len(wide)].to_numpy(), z["z"].iloc[len(wide):].to_numpy())
    assert abs(z["z"].mean()) < 0.1                                           # the league level is not counted as skill


def test_every_label_builds_its_half_tables_from_the_prediction_columns_alone():
    base = pitches(talent_sd=0.03, n_pitchers=30)
    pp = base.assign(is_in_zone=False, is_bip=True, is_weak=base["is_whiff"], chase_expected_full=0.25, weak_expected_full=0.25)[ic.PREDICTION_COLUMNS]

    for label in ic.LABELS:
        assert len(ic.half_tables(pp, label)) == 30


def test_the_summary_reports_the_mean_error_so_a_bias_is_visible():
    z = pd.DataFrame({"n_min": range(9), "z": [0.5, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5, 1.0]})

    assert ic.summarize(z)["z_mean"] == pytest.approx(1.0)


def test_a_label_with_too_few_qualified_pitcher_seasons_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(ic, "LABELS", ("whiff",))
    pp = pitches(talent_sd=0.03, n_pitchers=40)
    nobody = pd.DataFrame({"pitcher": range(40), "season": 2025, "qualified": False})

    with pytest.raises(ValueError, match="only 0 qualified pitcher-seasons"):
        ic.coverage(pp, nobody, {"whiff": CURVE})


def test_repeated_pitchers_are_collapsed_before_the_between_pitcher_variance_is_estimated(monkeypatch):
    seen = {}
    real = ic.rc.collapse_repeated_pitchers

    def spy(pitcher, n, diff, sampling_var):
        seen["pitchers"] = len(pitcher)
        return real(pitcher, n, diff, sampling_var)

    monkeypatch.setattr(ic.rc, "collapse_repeated_pitchers", spy)
    wide = ic.half_tables(pitches(talent_sd=0.03, n_pitchers=60), "whiff")

    ic.predictive_z(wide, CURVE)

    assert seen["pitchers"] == 60


def test_the_summary_also_splits_by_scored_pitches_per_game_when_they_are_given():
    z = pd.DataFrame({"n_min": range(9), "m_min": [9, 8, 7, 6, 5, 4, 3, 2, 1], "z": [-1, 1, 0, -2, 2, 0, -3, 3, 0]})

    summary = ic.summarize(z)

    assert summary["z_sd_by_sample_third"] == [1.0, 2.0, 3.0]
    assert summary["z_sd_by_cluster_third"] == [3.0, 2.0, 1.0]          # the order reverses because m_min runs the other way


def test_predictive_z_carries_the_smaller_halfs_pitches_per_game():
    wide = ic.half_tables(pitches(talent_sd=0.03, n_pitchers=40), "whiff")

    z = ic.predictive_z(wide, CURVE)

    assert z["m_min"].between(9, 11).all()                                 # 10 swings per game in the fixture
