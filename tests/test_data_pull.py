import sys
import types

import pandas as pd
import pytest

import data_pull as dp


def season_frame(seasons, rows_per_season=2, extra=None):
    frame = pd.DataFrame({"season": [s for s in seasons for _ in range(rows_per_season)],
                          "value": range(len(seasons) * rows_per_season)})
    return frame if extra is None else frame.assign(**extra)


def test_a_partial_pull_replaces_only_its_own_seasons_and_keeps_the_rest_in_season_order(tmp_path):
    path = tmp_path / "pitches.csv"
    season_frame([2025, 2026]).assign(value=[10, 11, 12, 13]).to_csv(path, index=False)

    out = dp.combine_with_existing(str(path), [season_frame([2024], extra={"value": [1, 2]})], [2024])

    assert out["season"].tolist() == [2024, 2024, 2025, 2025, 2026, 2026]
    assert out["value"].tolist() == [1, 2, 10, 11, 12, 13]


def test_pulling_a_season_twice_does_not_duplicate_its_rows(tmp_path):
    path = tmp_path / "pitches.csv"
    season_frame([2024, 2025, 2026]).to_csv(path, index=False)

    out = dp.combine_with_existing(str(path), [season_frame([2025], rows_per_season=3)], [2025])

    assert (out["season"] == 2025).sum() == 3 and (out["season"] == 2024).sum() == 2 and len(out) == 7


def test_a_full_pull_ignores_whatever_is_on_disk(tmp_path):
    path = tmp_path / "pitches.csv"
    pd.DataFrame({"season": [2020], "value": [99]}).to_csv(path, index=False)

    out = dp.combine_with_existing(str(path), [season_frame(dp.YEARS)], list(dp.YEARS))

    assert 2020 not in set(out["season"]) and set(out["season"]) == set(dp.YEARS)


class FakeResponse:
    def __init__(self, text):
        self.content = text.encode("utf-8")

    def raise_for_status(self):
        return None


def fake_requests(monkeypatch, text):
    module = types.SimpleNamespace(get=lambda url, headers=None, timeout=None: FakeResponse(text))
    monkeypatch.setitem(sys.modules, "requests", module)


@pytest.mark.parametrize("reply", ["", "   \n", "<!DOCTYPE html><html>sign in</html>", "a,b\n"])
def test_a_leaderboard_reply_with_no_data_raises_instead_of_passing_for_an_empty_season(monkeypatch, reply):
    fake_requests(monkeypatch, reply)

    with pytest.raises(ValueError):
        dp.read_leaderboard_csv("https://example.test/leaderboard", "arm angle")


def test_a_real_leaderboard_reply_is_parsed(monkeypatch):
    fake_requests(monkeypatch, "id,on_time_percent\n1,52.5\n2,48.0\n")

    frame = dp.read_leaderboard_csv("https://example.test/leaderboard", "swing timing")

    assert frame["on_time_percent"].tolist() == [52.5, 48.0]


def test_the_cache_is_only_wiped_when_the_directory_looks_like_a_pybaseball_cache(monkeypatch, tmp_path):
    victim = tmp_path / "my_data"
    victim.mkdir()
    (victim / "keep.txt").write_text("important")
    fake_pb = types.SimpleNamespace(cache=types.SimpleNamespace(config=types.SimpleNamespace(cache_directory=str(victim)),
                                                                enable=lambda: None))
    monkeypatch.setitem(sys.modules, "pybaseball", fake_pb)

    with pytest.raises(SystemExit):
        dp.reset_cache()

    assert (victim / "keep.txt").exists()
    cache = tmp_path / ".pybaseball" / "cache"
    cache.mkdir(parents=True)
    (cache / "old.json").write_text("{}")
    fake_pb.cache.config.cache_directory = str(cache)
    dp.reset_cache()
    assert not cache.exists()
