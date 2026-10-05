import numpy as np
import pandas as pd
import pytest

import merge_data


def statcast_file(tmp_path):
    rows = []
    # pitcher 1 and pitcher 2 share a name; pitcher 1 pitched for one team, pitcher 2 for three in 2025
    for pitcher, name, season, game_type, side, home, away in [
        (1, "García, Luis", 2025, "R", "Top", "HOU", "TEX"),        # pitching for the home team, HOU
        (2, "Garcia, Luis", 2025, "R", "Bot", "LAD", "WSH"),        # pitching for the away team, WSH
        (2, "Garcia, Luis", 2025, "R", "Top", "LAD", "WSH"),        # LAD
        (2, "Garcia, Luis", 2025, "R", "Bot", "SF", "LAA"),         # LAA
        (2, "Garcia, Luis", 2025, "S", "Bot", "SD", "SEA"),         # spring training, ignored
        (3, "Smith, Wade", 2025, "R", "Top", "SF", "SD"),           # unique name, SF
    ]:
        rows.append({"pitcher": pitcher, "player_name": name, "season": season, "game_type": game_type,
                     "inning_topbot": side, "home_team": home, "away_team": away})
    path = tmp_path / "statcast.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_the_name_crosswalk_keeps_every_pitcher_who_shares_a_normalized_name(tmp_path):
    crosswalk = merge_data.build_name_crosswalk(statcast_file(tmp_path))

    assert crosswalk["luis garcia"] == [1, 2]
    assert crosswalk["wade smith"] == [3]


def test_pitcher_season_teams_lists_each_regular_season_team_the_pitcher_threw_for(tmp_path):
    teams = merge_data.pitcher_season_teams(statcast_file(tmp_path))

    lookup = {(r.pitcher, r.season): r.teams for r in teams.itertuples()}
    assert lookup[(1, 2025)] == frozenset({"HOU"})
    assert lookup[(2, 2025)] == frozenset({"WSH", "LAD", "LAA"})      # spring training is not counted
    assert lookup[(3, 2025)] == frozenset({"SF"})


def fangraphs_rows(rows):
    return pd.DataFrame(rows, columns=["name", "season", "team"])


def resolve(tmp_path, rows):
    path = statcast_file(tmp_path)
    return merge_data.resolve_fangraphs_pitchers(
        fangraphs_rows(rows), merge_data.build_name_crosswalk(path), merge_data.pitcher_season_teams(path))


def test_a_unique_name_resolves_whatever_the_team_label_says(tmp_path):
    ids = resolve(tmp_path, [("Wade Smith", 2025, "SFG"), ("Wade Smith", 2025, "2 Tms"), ("Nobody Here", 2025, "SFG")])

    assert ids.iloc[0] == 3 and ids.iloc[1] == 3
    assert np.isnan(ids.iloc[2])


def test_a_shared_name_resolves_by_the_team_the_row_names(tmp_path):
    ids = resolve(tmp_path, [("Luis Garcia", 2025, "HOU"), ("Luis García", 2025, "WSN")])

    assert ids.tolist() == [1, 2]                                    # FanGraphs WSN is Statcast WSH


def test_a_shared_name_with_a_multi_team_row_resolves_to_the_pitcher_with_that_many_teams(tmp_path):
    ids = resolve(tmp_path, [("Luis García", 2025, "3 Tms")])

    assert ids.tolist() == [2]


def test_a_shared_name_that_cannot_be_told_apart_is_left_unmatched_instead_of_guessed(tmp_path):
    ids = resolve(tmp_path, [("Luis García", 2025, "2 Tms"),      # nobody threw for two teams
                             ("Luis García", 2025, "BOS"),       # nobody threw for BOS
                             ("Luis García", 2024, "HOU")])      # no team data for that season

    assert ids.isna().all()


def test_fangraphs_team_labels_map_to_statcast_abbreviations():
    assert merge_data.statcast_team("SFG") == "SF"
    assert merge_data.statcast_team("CHW") == "CWS"
    assert merge_data.statcast_team("HOU") == "HOU"


def pitch_file(tmp_path, rows):
    path = tmp_path / "pitches.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_sequencing_features_count_only_regular_season_pitches(tmp_path):
    base = {"game_pk": 1, "pitcher": 10, "at_bat_number": 1, "season": 2025}
    path = pitch_file(tmp_path, [
        {**base, "game_type": "R", "pitch_number": 1, "pitch_type": "FF", "release_speed": 95.0},
        {**base, "game_type": "R", "pitch_number": 2, "pitch_type": "SL", "release_speed": 85.0},
        # a spring-training at-bat with a repeat and a tiny gap must not enter the averages
        {**base, "game_pk": 2, "game_type": "S", "pitch_number": 1, "pitch_type": "FF", "release_speed": 95.0},
        {**base, "game_pk": 2, "game_type": "S", "pitch_number": 2, "pitch_type": "FF", "release_speed": 95.0},
    ])

    out = merge_data.compute_sequencing_features(path)

    row = out[out["pitch_type"] == "SL"].iloc[0]
    assert row["avg_velocity_gap_from_prev"] == pytest.approx(10.0)
    assert row["repeat_pct"] == 0.0
    assert out[out["pitch_type"] == "FF"].empty                      # the only FF with a previous pitch was in spring training


def test_pitch_characteristics_average_only_regular_season_pitches(tmp_path):
    rows = []
    for game_type, speed in (("R", 90.0), ("R", 92.0), ("S", 70.0)):
        rows.append({"pitcher": 10, "pitch_type": "FF", "season": 2025, "game_type": game_type, "release_speed": speed,
                     **{c: 1.0 for c in merge_data.CHAR_COLS if c != "release_speed"}})
    path = pitch_file(tmp_path, rows)

    out = merge_data.compute_pitch_characteristics(path)

    assert out["release_speed_mean"].iloc[0] == pytest.approx(91.0)


def test_names_normalize_to_lowercase_ascii_without_punctuation():
    assert merge_data.normalize_name("José  Berríos") == "jose berrios"
    assert merge_data.normalize_name("J.T. Brubaker") == "jt brubaker"
    assert merge_data.normalize_name("Jean-Carlos O'Neil") == "jean carlos oneil"


def test_pitch_mix_entropy_is_zero_for_one_pitch_and_one_bit_for_an_even_split():
    rates = pd.DataFrame({"pitcher": [1, 2, 2], "season": 2025, "pitches": [100, 50, 50]})

    out = merge_data.compute_pitch_mix_entropy(rates).set_index("pitcher")["pitch_mix_entropy"]

    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(1.0)


def test_covariate_rows_for_seasons_that_were_not_pulled_are_dropped():
    covariates = pd.DataFrame({"pitcher": [1, 1, 2], "season": [2023, 2024, 2023], "stuff_plus": [100.0, 110.0, 90.0]})

    out = merge_data.restrict_to_pulled_seasons(covariates, [2024, 2025, 2026])

    assert out["season"].tolist() == [2024] and out["pitcher"].tolist() == [1]
