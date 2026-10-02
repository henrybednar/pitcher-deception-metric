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
