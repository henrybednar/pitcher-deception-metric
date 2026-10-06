import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from driver_features import fdr_adjusted_p_values


def noisy_frame(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    return X, rng


def test_fdr_adjusted_p_values_flag_the_real_effect_and_not_the_noise_columns():
    X, rng = noisy_frame()
    y = 0.5 * X["f0"] + rng.normal(size=len(X))

    q = fdr_adjusted_p_values(X, y)

    assert q["f0"] < 0.001
    assert (q.drop("f0") > 0.05).all()


def test_fdr_adjusted_p_values_are_never_below_the_raw_p_values():
    X, rng = noisy_frame(seed=1)
    y = 0.1 * X["f1"] + rng.normal(size=len(X))

    q = fdr_adjusted_p_values(X, y)
    raw = sm.OLS(y, sm.add_constant(X)).fit().pvalues[X.columns]

    assert (q >= raw - 1e-12).all()
    assert list(q.index) == list(X.columns)


def test_fdr_adjusted_p_values_do_not_depend_on_the_scale_of_a_feature():
    X, rng = noisy_frame(seed=2)
    y = 0.3 * X["f2"] + rng.normal(size=len(X))
    rescaled = X.assign(f2=X["f2"] * 1000)

    assert fdr_adjusted_p_values(rescaled, y)["f2"] == pytest.approx(fdr_adjusted_p_values(X, y)["f2"])


def test_clustered_p_values_are_larger_than_naive_ones_when_each_pitcher_has_two_identical_rows():
    X, rng = noisy_frame(n=300, seed=3)
    y = 0.12 * X["f3"] + rng.normal(size=len(X))
    groups = np.arange(len(X))
    # every pitcher appears twice with the same values, as a pitcher's two seasons can be strongly dependent
    X2, y2, g2 = pd.concat([X, X], ignore_index=True), pd.concat([y, y], ignore_index=True), np.concatenate([groups, groups])

    naive = fdr_adjusted_p_values(X2, y2)
    clustered = fdr_adjusted_p_values(X2, y2, groups=g2)

    assert clustered["f3"] > naive["f3"]            # the duplicated rows overstate the evidence for the naive fit


def test_clustered_p_values_match_the_single_row_fit_when_pitchers_are_not_repeated():
    X, rng = noisy_frame(n=800, seed=4)
    y = 0.05 * X["f1"] + rng.normal(size=len(X))

    clustered = fdr_adjusted_p_values(X, y, groups=np.arange(len(X)))
    naive = fdr_adjusted_p_values(X, y)

    assert clustered.to_numpy() == pytest.approx(naive.to_numpy(), abs=0.05)   # one row per cluster is the robust fit


def test_load_columns_keeps_regular_season_rows_and_drops_the_game_type_column(tmp_path):
    from driver_features import load_columns

    path = tmp_path / "pitches.csv"
    pd.DataFrame({"pitcher": [1, 2, 3], "game_type": ["R", "S", "L"], "spin_axis": [200.0, 190.0, 180.0]}).to_csv(path, index=False)

    out = load_columns(str(path), ["pitcher", "spin_axis"])

    assert out["pitcher"].tolist() == [1]
    assert "game_type" not in out.columns


def test_weighted_std_weights_each_pitch_type_by_its_pitches():
    from driver_features import weighted_std

    assert weighted_std(np.array([10.0, 20.0]), np.array([1.0, 1.0])) == pytest.approx(5.0)
    assert weighted_std(np.array([10.0, 20.0]), np.array([9.0, 1.0])) == pytest.approx(3.0)
    assert np.isnan(weighted_std(np.array([10.0]), np.array([5.0])))


def test_cross_pitch_features_skip_pitchers_with_one_pitch_type_and_thin_pitch_types():
    from driver_features import build_cross_pitch_features

    pitch_types = pd.DataFrame({
        "pitcher": [1, 1, 2, 3, 3], "season": 2025, "pitches": [100, 100, 300, 200, 5],
        "arm_angle_mean": [40.0, 50.0, 45.0, 30.0, 60.0], "release_pos_x_mean": [1.0, 1.0, 1.0, 1.0, 5.0],
        "release_pos_z_mean": [5.0, 5.2, 5.0, 5.0, 4.0], "vaa_mean": [-4.0, -5.0, -4.5, -4.0, -8.0],
        "avg_velocity_gap_per_switch": [5.0, 7.0, 6.0, 4.0, 9.0], "n_switched": [10, 30, 50, 40, 2], "repeat_pct": [0.4, 0.2, 0.5, 0.3, 0.1],
        "release_extension_mean": [6.0, 6.2, 6.1, 6.0, 5.0],
    })

    out = build_cross_pitch_features(pitch_types).set_index("pitcher")

    assert list(out.index) == [1]                               # pitcher 2 has one type, pitcher 3's second type has under 20 pitches
    assert out.loc[1, "n_pitch_types"] == 2
    assert out.loc[1, "arm_angle_cross_pitch_std"] == pytest.approx(5.0)
    assert out.loc[1, "velocity_gap_per_switch"] == pytest.approx(6.5)     # (5 x 10 + 7 x 30) / 40, weighted by switches


def test_pooled_gap_per_switch_skips_types_with_no_switches_and_is_nan_when_none_switched():
    from driver_features import pooled_gap_per_switch

    mixed = pd.DataFrame({"avg_velocity_gap_per_switch": [4.0, np.nan, 10.0], "n_switched": [30, 0, 10]})
    none = pd.DataFrame({"avg_velocity_gap_per_switch": [np.nan, np.nan], "n_switched": [0, 0]})

    assert pooled_gap_per_switch(mixed) == pytest.approx(5.5)            # (4 x 30 + 10 x 10) / 40
    assert np.isnan(pooled_gap_per_switch(none))
