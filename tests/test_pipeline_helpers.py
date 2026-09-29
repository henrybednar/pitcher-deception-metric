import numpy as np
import pandas as pd
import pytest

import build_pitch_table
import fit_full_model
import pitch_hygiene
from export_site_stats import pitcher_roles
from fit_full_model import LOCATION_FEATURES, STUFF_FEATURES, fit_tier, recalibrate_oof_isotonic


def aligned_frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "pitcher": range(n), "pitch_type": ["FF"] * n, "season": [2025] * n,
        "game_pk": range(100, 100 + n), "half": [0, 1] * (n // 2),
    })


@pytest.fixture
def predictions_file(tmp_path, monkeypatch):
    path = tmp_path / "per_pitch_predictions.csv"
    monkeypatch.setattr(build_pitch_table, "PREDICTIONS_FILE", str(path))
    return path


def test_read_aligned_predictions_returns_the_file_when_rows_match(predictions_file):
    df = aligned_frame()
    df.to_csv(predictions_file, index=False)

    existing = build_pitch_table.read_aligned_predictions(df)

    assert len(existing) == len(df)


def test_read_aligned_predictions_refuses_a_different_row_order(predictions_file):
    df = aligned_frame()
    df.iloc[::-1].to_csv(predictions_file, index=False)

    with pytest.raises(AssertionError, match="misaligned"):
        build_pitch_table.read_aligned_predictions(df)


def test_read_aligned_predictions_refuses_a_different_row_count(predictions_file):
    df = aligned_frame()
    df.iloc[:-1].to_csv(predictions_file, index=False)

    with pytest.raises(AssertionError, match="row count"):
        build_pitch_table.read_aligned_predictions(df)


def test_read_aligned_predictions_treats_missing_pitch_types_as_equal(predictions_file):
    df = aligned_frame()
    df.loc[1, "pitch_type"] = np.nan
    df.to_csv(predictions_file, index=False)

    build_pitch_table.read_aligned_predictions(df)


def test_save_predictions_appends_columns_by_position(predictions_file):
    df = aligned_frame()
    df.to_csv(predictions_file, index=False)
    existing = build_pitch_table.read_aligned_predictions(df)
    df["new_col"] = [0.1, 0.2, 0.3, 0.4]

    build_pitch_table.save_predictions(existing, df, ["new_col"])

    assert pd.read_csv(predictions_file)["new_col"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_pitcher_roles_labels_starters_relievers_and_swingmen():
    games = {(1, 10): 90, (1, 11): 95, (2, 20): 15, (2, 21): 20, (3, 30): 45, (3, 31): 40}
    raw = pd.DataFrame(
        [(pitcher, 2025, game) for (pitcher, game), pitches in games.items() for _ in range(pitches)],
        columns=["pitcher", "season", "game_pk"],
    )

    roles = pitcher_roles(raw).set_index("pitcher")

    assert roles["role"].to_dict() == {1: "SP", 2: "RP", 3: "MR"}
    assert roles.loc[1, "med_pitches_per_app"] == 92.5


def test_pitcher_roles_prefers_real_games_started_share_over_the_pitch_count_proxy():
    # Pitcher 1's own appearances average a starter-like 90 pitches, but he actually started only
    # 2 of 20 games (a swingman who happened to throw long in his rare starts) — real usage should
    # override the proxy's "SP" guess.
    games = {(1, 10): 90, (1, 11): 95}
    raw = pd.DataFrame(
        [(pitcher, 2025, game) for (pitcher, game), pitches in games.items() for _ in range(pitches)],
        columns=["pitcher", "season", "game_pk"],
    )
    usage = pd.DataFrame({"pitcher": [1], "season": [2025], "games": [20], "games_started": [2]})

    roles = pitcher_roles(raw, usage=usage).set_index("pitcher")

    assert roles.loc[1, "role"] == "MR"


def test_pitcher_roles_falls_back_to_the_proxy_when_usage_is_missing_for_a_pitcher():
    games = {(1, 10): 90, (1, 11): 95, (2, 20): 15, (2, 21): 20}
    raw = pd.DataFrame(
        [(pitcher, 2025, game) for (pitcher, game), pitches in games.items() for _ in range(pitches)],
        columns=["pitcher", "season", "game_pk"],
    )
    usage = pd.DataFrame({"pitcher": [1], "season": [2025], "games": [20], "games_started": [18]})

    roles = pitcher_roles(raw, usage=usage).set_index("pitcher")

    assert roles.loc[1, "role"] == "SP"    # from real usage
    assert roles.loc[2, "role"] == "RP"    # pitcher 2 has no usage row, falls back to the proxy


@pytest.fixture
def small_pitch_frame(monkeypatch):
    monkeypatch.setattr(fit_full_model, "MIN_N_FOR_MODEL", 100)
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({c: rng.normal(size=n) for c in STUFF_FEATURES + LOCATION_FEATURES if c != "p_throws"})
    df["p_throws"] = pd.Categorical(rng.choice(["L", "R"], n))
    df["pitcher"] = rng.integers(0, 30, n)
    df["pitch_type"] = "FF"
    df["target"] = df["release_speed"] * 2 + rng.normal(scale=0.1, size=n)
    return df


def test_fit_tier_without_tendencies_predicts_every_row_out_of_fold(small_pitch_frame):
    df = small_pitch_frame
    mask = pd.Series(True, index=df.index)

    fit_tier(df, "expected", "target", mask, "regress", STUFF_FEATURES + LOCATION_FEATURES, with_tendencies=False)

    assert df["expected"].notna().all()
    assert np.corrcoef(df["expected"], df["target"])[0, 1] > 0.9


def test_fit_tier_leaves_unscored_pitch_types_empty(small_pitch_frame):
    df = small_pitch_frame
    df.loc[:49, "pitch_type"] = np.nan
    mask = pd.Series(True, index=df.index)

    fit_tier(df, "expected", "target", mask, "regress", STUFF_FEATURES + LOCATION_FEATURES, with_tendencies=False)

    assert df.loc[:49, "expected"].isna().all()
    assert df.loc[50:, "expected"].notna().all()


def test_plate_height_is_normalized_to_the_zone_and_nan_for_a_broken_zone():
    frame = pd.DataFrame({"plate_z": [2.5, 2.5, 2.5, 2.5], "sz_bot": [1.5, 1.6, 1.6, np.nan], "sz_top": [3.5, 1.6, 1.0, 3.5]})

    result = build_pitch_table.normalize_plate_z(frame)

    assert result.iloc[0] == pytest.approx(0.5)
    assert result.iloc[1:].isna().all()       # zero height, negative height, missing bottom


def test_pitch_count_in_appearance_is_a_one_based_running_count_per_pitcher_game_regardless_of_row_order():
    # Rows arrive out of chronological order on purpose, matching a real CSV read that isn't
    # guaranteed to be pre-sorted; the count must follow at_bat_number/pitch_number, not row order.
    # A row id (not used by the function) lets each row be checked unambiguously afterward.
    frame = pd.DataFrame({
        "row": ["ab2pn1", "ab1pn2", "ab1pn1", "other_pitcher", "other_game"],
        "pitcher": [1, 1, 1, 2, 1],
        "game_pk": [100, 100, 100, 100, 200],
        "at_bat_number": [2, 1, 1, 1, 1],
        "pitch_number": [1, 2, 1, 1, 1],
    })

    out = build_pitch_table.add_pitch_count_in_appearance(frame).set_index("row")["pitch_count_in_appearance"]

    assert out["ab1pn1"] == 1     # first pitch thrown, this pitcher, this game
    assert out["ab1pn2"] == 2     # second pitch thrown
    assert out["ab2pn1"] == 3     # third pitch thrown
    assert out["other_pitcher"] == 1   # a different pitcher in the same game starts its own count
    assert out["other_game"] == 1      # the same pitcher in a different game starts its own count


def test_filter_regular_season_drops_postseason_and_spring_training_and_the_game_type_column():
    frame = pd.DataFrame({"pitcher": [1, 2, 3, 4], "game_type": ["R", "S", "D", "W"]})

    result = build_pitch_table.filter_regular_season(frame)

    assert result["pitcher"].tolist() == [1]
    assert "game_type" not in result.columns


def tracking_frame(n=3):
    frame = pd.DataFrame({c: [1.0] * n for c in pitch_hygiene.REQUIRED_TRACKING})
    frame["arm_angle"] = 30.0
    frame["pitch_type"] = "FF"
    return frame


def test_pitches_missing_core_tracking_are_left_out_of_scoring():
    frame = tracking_frame()
    frame.loc[1, "effective_speed"] = np.nan

    out = pitch_hygiene.blank_incomplete_tracking(frame)

    assert out["pitch_type"].isna().tolist() == [False, True, False]


def test_a_missing_arm_angle_alone_does_not_remove_a_pitch_from_scoring():
    frame = tracking_frame()
    frame["arm_angle"] = np.nan

    assert pitch_hygiene.blank_incomplete_tracking(frame)["pitch_type"].notna().all()


def test_impossible_zone_bounds_are_cleared_by_the_domain_gates():
    frame = pd.DataFrame({"sz_top": [3.4, 3.4, 9.0], "sz_bot": [1.6, 0.0, 1.6]})

    out, cleared = pitch_hygiene.apply_domain_gates(frame)

    assert out["sz_bot"].isna().tolist() == [False, True, False]
    assert out["sz_top"].isna().tolist() == [False, False, True]


def typed_frame(frame, counts):
    frame = frame.copy()
    labels = np.concatenate([[t] * n for t, n in counts.items()])
    frame["pitch_type"] = labels
    return frame


def test_a_rare_type_joins_its_neighbors_model_and_is_still_scored(small_pitch_frame):
    df = typed_frame(small_pitch_frame, {"CU": 500, "KC": 60, "FS": 40})
    mask = pd.Series(True, index=df.index)

    fit_tier(df, "expected", "target", mask, "regress", STUFF_FEATURES + LOCATION_FEATURES, with_tendencies=False)

    assert df.loc[df["pitch_type"] == "KC", "expected"].notna().all()     # pooled into CU
    assert df.loc[df["pitch_type"] == "CU", "expected"].notna().all()
    assert df.loc[df["pitch_type"] == "FS", "expected"].isna().all()      # too rare, no neighbor, pool too small


def test_rare_types_without_a_neighbor_share_an_other_model_when_the_pool_is_big_enough(small_pitch_frame):
    df = typed_frame(small_pitch_frame, {"CU": 400, "FS": 60, "SV": 60, "ST": 80})
    mask = pd.Series(True, index=df.index)

    fit_tier(df, "expected", "target", mask, "regress", STUFF_FEATURES + LOCATION_FEATURES, with_tendencies=False)

    assert df["expected"].notna().all()


def test_fit_tier_accepts_season_as_a_categorical_feature_across_folds(small_pitch_frame):
    df = small_pitch_frame
    rng = np.random.default_rng(1)
    df["season"] = rng.choice([2025, 2026], len(df))
    mask = pd.Series(True, index=df.index)

    fit_tier(df, "expected", "target", mask, "regress",
             STUFF_FEATURES + LOCATION_FEATURES + ["season"], with_tendencies=False)

    assert df["expected"].notna().all()
    assert np.corrcoef(df["expected"], df["target"])[0, 1] > 0.9


def test_recalibrate_oof_isotonic_corrects_a_tail_bias_without_leaking_a_pitchers_own_outcome():
    rng = np.random.default_rng(2)
    n = 3000
    pitcher = rng.integers(0, 60, n)
    expected = rng.uniform(3, 15, n)
    # A systematic bias only at the low end, mirroring the real finding: actual runs above
    # expected when expected is small, flat elsewhere.
    bias = np.where(expected < 5, 1.0, 0.0)
    actual = expected + bias + rng.normal(0, 0.3, n)
    sub = pd.DataFrame({"pitcher": pitcher, "actual": actual, "expected": expected})

    recal = recalibrate_oof_isotonic(sub, "actual", "expected")

    low = expected < 5
    assert np.abs((actual[low] - expected[low]).mean() - 1.0) < 0.1          # bias is really there
    assert np.abs((actual[low] - recal[low]).mean()) < 0.15                  # recalibration removes it
    assert np.abs((actual[~low] - recal[~low]).mean()) < 0.1                # unbiased region stays unbiased
    assert not np.isnan(recal).any()


def test_fit_tier_still_works_when_a_fold_sees_only_one_season(small_pitch_frame):
    # GroupKFold groups by pitcher, so a fold's test rows could in principle be dominated by
    # pitchers from a single season; the fixed category set must not assume both are present
    # in every fold's train/test split.
    df = small_pitch_frame
    df["season"] = np.where(df["pitcher"] < 20, 2025, 2026)
    mask = pd.Series(True, index=df.index)

    fit_tier(df, "expected", "target", mask, "regress",
             STUFF_FEATURES + LOCATION_FEATURES + ["season"], with_tendencies=False)

    assert df["expected"].notna().all()


def test_season_feature_absorbs_a_season_level_calibration_gap_at_the_population_level(small_pitch_frame):
    # A single held-out row is not enough to tell whether the season feature is doing its
    # calibration job: a tree ensemble doesn't split on every feature along every path, so one
    # probe row can show a flat 0.0 gap even when season is absorbing a real population-level
    # shift everywhere else. The real audit hit exactly this with a single-row probe before
    # rerunning across every held-out row. This checks the population mean instead.
    df = small_pitch_frame
    rng = np.random.default_rng(3)
    df["season"] = rng.choice([2025, 2026], len(df))
    season_shift = np.where(df["season"] == 2025, 5.0, -5.0)
    df["target"] = df["target"] + season_shift
    mask = pd.Series(True, index=df.index)

    fit_tier(df, "expected", "target", mask, "regress",
             STUFF_FEATURES + LOCATION_FEATURES + ["season"], with_tendencies=False)

    by_season = df.groupby("season")["expected"].mean()
    assert (by_season[2025] - by_season[2026]) > 5.0
