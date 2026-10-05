"""
Pitcher Deception Project — Data Pull
======================================
Pulls and merges everything needed for the deception metric:
  1. Raw pitch-level Statcast data (2024-2026): velo, spin, movement,
     release points, extension, VAA/HAA (computed)
  2. Outcome rates by pitcher/pitch type: Whiff%, Chase%, GB%, Strike%
     (computed from #1)
  3. Swing timing / miss distance leaderboard (bat tracking, 2024+ only)
  4. Arm angle leaderboard (tunneling feature)
  5. Pitch tempo leaderboard (rhythm disruption)

Stuff+ / Location+ / PitchingBot comes from a FanGraphs export dropped in by
hand (see merge_data.merge_fangraphs_stuff) — FanGraphs blocks scripted
access with a 403, so there is no automated pull for it here.

Run: `pip install -r requirements-pull.txt` first (pybaseball and requests; the pinned versions the pipeline was run with).

    python data_pull.py                 pull every season in YEARS
    python data_pull.py --years 2024    pull only those seasons and keep the other seasons' rows from the
                                        files already in raw/ (so adding a season does not re-pull the rest)
"""

import argparse
import io
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# pybaseball and requests are imported where they are used, so importing this module (for its helpers, or in a
# test) neither needs them installed nor touches the cache directory.


def reset_cache() -> None:
    """Empty pybaseball's cache directory and turn the cache back on for this pull.

    The cache keys on each call's own arguments (the date range, the year), not on when it ran. A re-pull with the
    same SEASON_RANGES as last time would otherwise silently return last time's stale result instead of the new
    games, so every pull starts from an empty cache directory. pybaseball's own cache.purge() reads every existing
    record to remove it and raises JSONDecodeError on a record left truncated by a killed process, so the directory
    is wiped directly instead. It refuses a directory that does not look like a pybaseball cache, since the location
    comes from the PYBASEBALL_CACHE environment variable and a mistaken value must not delete something else."""
    import pybaseball as pb

    cache_dir = Path(pb.cache.config.cache_directory)
    if "pybaseball" not in str(cache_dir).lower():
        raise SystemExit(f"refusing to wipe {cache_dir}: it does not look like a pybaseball cache directory")
    shutil.rmtree(cache_dir, ignore_errors=True)
    pb.cache.enable()  # avoids re-downloading on repeat runs within this one pull

YEARS = [2024, 2025, 2026]
SEASON_RANGES = {
    2024: ("2024-03-20", "2024-11-01"),  # bat tracking is complete from April 2024, so every component can be scored
    2025: ("2025-03-18", "2025-11-01"),
    2026: ("2026-03-24", "2026-11-01"),  # 2026-11-01 safely postdates the season; no need to narrow it
}

HEADERS = {"User-Agent": "Mozilla/5.0"}  # Savant 403s without a UA header


# ---------------------------------------------------------------------------
# 1. Raw pitch-level Statcast data
# ---------------------------------------------------------------------------
def pull_statcast_pitches(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Full pitch-level pull. pybaseball auto-chunks by date under the hood."""
    import pybaseball as pb

    return pb.statcast(start_dt=start_dt, end_dt=end_dt)


def add_vaa_haa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vertical/Horizontal Approach Angle at the front of home plate.
    Not provided by Statcast directly — derived from the pitch's
    kinematic trajectory (constant acceleration model Statcast publishes).
    Formula follows the standard public derivation used by Savant/Alan Nathan.
    """
    df = df.copy()
    y0 = 50.0        # Statcast tracking start, ft from home plate
    yf = 17 / 12      # front of home plate, ft

    vy_f = -np.sqrt(df["vy0"] ** 2 - 2 * df["ay"] * (y0 - yf))
    t = (vy_f - df["vy0"]) / df["ay"]
    vz_f = df["vz0"] + df["az"] * t
    vx_f = df["vx0"] + df["ax"] * t

    df["vaa"] = -np.degrees(np.arctan(vz_f / vy_f))
    df["haa"] = -np.degrees(np.arctan(vx_f / vy_f))
    return df


def add_movement_inches(df: pd.DataFrame) -> pd.DataFrame:
    """pfx_x / pfx_z come back in feet from pybaseball — convert to inches
    to match Savant's IVB (induced vertical break) and HB (horizontal break)."""
    df = df.copy()
    df["ivb_in"] = df["pfx_z"] * 12
    df["hb_in"] = df["pfx_x"] * 12
    return df


# ---------------------------------------------------------------------------
# 2. Outcome rates computed from pitch-level data
# ---------------------------------------------------------------------------
SWING_DESCRIPTIONS = {
    "foul", "foul_tip", "hit_into_play", "swinging_strike",
    "swinging_strike_blocked", "missed_bunt", "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
IN_ZONE = set(range(1, 10))  # Savant zones 1-9 = in zone, 11-14 = out of zone


def compute_outcome_rates(df: pd.DataFrame, group_cols=("pitcher", "pitch_type")) -> pd.DataFrame:
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["is_in_zone"] = df["zone"].isin(IN_ZONE)
    df["is_chase"] = df["is_swing"] & ~df["is_in_zone"]
    df["is_strike"] = df["type"].isin(["S", "X"])
    df["is_gb"] = df["bb_type"] == "ground_ball"
    df["is_batted_ball"] = df["bb_type"].notna()

    grouped = df.groupby(list(group_cols)).agg(
        pitches=("pitch_type", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        pitches_out_zone=("is_in_zone", lambda s: (~s).sum()),
        chases=("is_chase", "sum"),
        strikes=("is_strike", "sum"),
        batted_balls=("is_batted_ball", "sum"),
        ground_balls=("is_gb", "sum"),
    ).reset_index()

    grouped["whiff_pct"] = grouped["whiffs"] / grouped["swings"]
    grouped["chase_pct"] = grouped["chases"] / grouped["pitches_out_zone"]
    grouped["strike_pct"] = grouped["strikes"] / grouped["pitches"]
    grouped["gb_pct"] = grouped["ground_balls"] / grouped["batted_balls"]
    return grouped


# ---------------------------------------------------------------------------
# 3. Swing timing / miss distance (bat tracking, 2024+, pitcher view)
# ---------------------------------------------------------------------------
def read_leaderboard_csv(url: str, name: str) -> pd.DataFrame:
    """One Savant leaderboard as a frame. An empty reply or an HTML page (an error or sign-in page served with a
    200) raises, so a season that did not come back can never pass for a season with no rows."""
    import requests

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode("utf-8")
    if not text.strip() or text.strip().startswith("<!"):
        raise ValueError(f"{name} pull returned no CSV; verify the URL params against the live page")
    frame = pd.read_csv(io.StringIO(text))
    if frame.empty:
        raise ValueError(f"{name} pull returned a CSV with no rows")
    return frame


def pull_swing_timing(year: int, min_swings="1") -> pd.DataFrame:
    """Params confirmed live 2026-09-14 by watching the leaderboard's own
    Download CSV request in the network tab — the endpoint uses a season[]
    year, not a date range, and different field names than the old guess."""
    url = (
        "https://baseballsavant.mlb.com/leaderboard/bat-tracking/swing-timing-miss-distance"
        f"?type=pitcher&season[]={year}&splitYear=1"
        f"&min={min_swings}&minSplit={min_swings}"
        "&gameType[]=R&dateStart=&dateEnd=&batSide=&contactType=&attackZone=&pitchHand="
        "&csv=true"
    )
    return read_leaderboard_csv(url, "swing timing")


# ---------------------------------------------------------------------------
# 4. Arm angle (tunneling feature — actual release angle, not inferred)
# ---------------------------------------------------------------------------
def pull_arm_angle(year: int, min_pitches="1") -> pd.DataFrame:
    """Params confirmed live 2026-09-14 by watching the leaderboard's own
    Download CSV request in the network tab — key is `season`, not `year`."""
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitcher-arm-angles"
        f"?batSide=&dateStart=&dateEnd=&gameType=R&groupBy="
        f"&min={min_pitches}&minGroupPitches={min_pitches}"
        f"&perspective=back&pitchHand=&pitchType=&season={year}"
        "&size=small&sort=ascending&team=&csv=true"
    )
    return read_leaderboard_csv(url, "arm angle")


# ---------------------------------------------------------------------------
# 5. Pitch tempo (rhythm disruption — a real deception lever, not just pace)
# ---------------------------------------------------------------------------
def pull_pitch_tempo(year: int, min_pitches="1") -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-tempo"
        f"?type=Pit&season_start={year}&season_end={year}"
        f"&n={min_pitches}&game_type=Regular&split=no&with_team_only=1&csv=true"
    )
    return read_leaderboard_csv(url, "pitch tempo")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
OUTPUT_FILES = {
    "pitches": "raw/statcast_pitch_level_2024_2026.csv",
    "rates": "raw/outcome_rates_by_pitcher_pitchtype_2024_2026.csv",
    "swing_timing": "raw/swing_timing_by_pitcher_2024_2026.csv",
    "arm_angle": "raw/arm_angle_by_pitcher_2024_2026.csv",
    "tempo": "raw/pitch_tempo_by_pitcher_2024_2026.csv",
}


def combine_with_existing(filename: str, frames: list[pd.DataFrame], pulled_years: list[int]) -> pd.DataFrame:
    """The pulled seasons' rows, plus every other season's rows from the file already on disk when only some
    seasons were pulled. Rows end up in season order, like a full pull."""
    new = pd.concat(frames, ignore_index=True)
    if set(pulled_years) >= set(YEARS) or not Path(filename).exists():
        return new
    old = pd.read_csv(filename, low_memory=False)
    old = old[~old["season"].isin(pulled_years)]
    return pd.concat([old, new], ignore_index=True, sort=False).sort_values("season", kind="stable", ignore_index=True)


def pull_season(year: int) -> dict[str, pd.DataFrame]:
    """Every source for one season, keyed like OUTPUT_FILES. Any source that cannot be pulled raises, so a season is
    either complete or the whole pull stops before a file is touched."""
    start, end = SEASON_RANGES[year]
    print(f"--- {year} ---")

    print("pulling raw pitch-level statcast...")
    pitches = add_movement_inches(add_vaa_haa(pull_statcast_pitches(start, end)))
    pitches["season"] = year

    print("computing outcome rates...")
    rates = compute_outcome_rates(pitches[pitches["game_type"] == "R"])   # regular season only, like the scores
    rates["season"] = year

    frames = {"pitches": pitches, "rates": rates}
    for key, label, puller in (("swing_timing", "swing timing", pull_swing_timing), ("arm_angle", "arm angle", pull_arm_angle),
                               ("tempo", "pitch tempo", pull_pitch_tempo)):
        print(f"pulling {label} leaderboard...")
        try:
            frames[key] = puller(year).assign(season=year)
        except Exception as e:
            raise RuntimeError(f"{label} pull for {year} failed: {e}") from e
        time.sleep(1)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=YEARS, choices=YEARS,
                        help="seasons to pull; the other seasons' rows are kept from the existing files")
    pulled_years = sorted(set(parser.parse_args().years))
    reset_cache()

    pulled = {}
    try:
        for year in pulled_years:
            for key, frame in pull_season(year).items():
                pulled.setdefault(key, []).append(frame)
    except Exception as e:
        # nothing has been written yet, so the files in raw/ are exactly as they were
        sys.exit(f"Pull failed, no file was changed: {e}")

    Path("raw").mkdir(exist_ok=True)
    for key, filename in OUTPUT_FILES.items():
        if set(pulled_years) < set(YEARS) and not Path(filename).exists():
            print(f"Warning: {filename} did not exist, so it now holds only {pulled_years}.")
        combined = combine_with_existing(filename, pulled[key], pulled_years)
        missing = set(YEARS if set(pulled_years) >= set(YEARS) else pulled_years) - set(combined["season"].unique())
        if missing:
            sys.exit(f"{filename} would be missing seasons {sorted(missing)}; nothing was written for it.")
        combined.to_csv(filename, index=False)
    print(f"Done. {len(OUTPUT_FILES)}/{len(OUTPUT_FILES)} CSVs written: {list(OUTPUT_FILES.values())}")


if __name__ == "__main__":
    main()
