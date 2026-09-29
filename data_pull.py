"""
Pitcher Deception Project — Data Pull
======================================
Pulls and merges everything needed for the deception metric:
  1. Raw pitch-level Statcast data (2025-2026): velo, spin, movement,
     release points, extension, VAA/HAA (computed)
  2. Outcome rates by pitcher/pitch type: Whiff%, Chase%, GB%, Strike%
     (computed from #1)
  3. Swing timing / miss distance leaderboard (bat tracking, 2024+ only)
  4. Stuff+ / Location+ / PitchingBot, by pitch type, from FanGraphs
  5. Arm angle leaderboard (tunneling feature)
  6. Pitch tempo leaderboard (rhythm disruption)

Run: `pip install pybaseball pandas numpy requests --upgrade` first
(need pybaseball >=2.2.7 for Stuff+ support).
"""

import io
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import pybaseball as pb

# The cache keys on each call's own arguments (the date range, the year), not on when it ran. A
# re-pull with the same SEASON_RANGES as last time would otherwise silently return last time's
# stale result instead of the new games, so every run starts from an empty cache directory.
# pybaseball's own cache.purge() reads every existing record to remove it and raises
# JSONDecodeError on a record left truncated by a killed process, so the directory is wiped
# directly instead: same effect, and it can't be broken by a bad record.
shutil.rmtree(pb.cache.config.cache_directory, ignore_errors=True)
pb.cache.enable()  # avoids re-downloading on repeat runs within this one pull

YEARS = [2025, 2026]
SEASON_RANGES = {
    2025: ("2025-03-18", "2025-11-01"),
    2026: ("2026-03-24", "2026-11-01"),  # 2026-11-01 safely postdates the season; no need to narrow it
}

HEADERS = {"User-Agent": "Mozilla/5.0"}  # Savant 403s without a UA header


# ---------------------------------------------------------------------------
# 1. Raw pitch-level Statcast data
# ---------------------------------------------------------------------------
def pull_statcast_pitches(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Full pitch-level pull. pybaseball auto-chunks by date under the hood."""
    df = pb.statcast(start_dt=start_dt, end_dt=end_dt)
    return df


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
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode("utf-8")
    if not text.strip() or text.strip().startswith("<!"):
        raise ValueError("Swing timing pull returned no CSV — verify URL params against the live page.")
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------
# 4. Stuff+ / Location+ / PitchingBot by pitch type (FanGraphs)
# ---------------------------------------------------------------------------
def pull_stuff_plus(year: int) -> pd.DataFrame:
    """
    stat_columns='ALL' pulls every FanGraphs column, including the
    Stuff+/Location+/PitchingBot-by-pitch-type columns confirmed present
    in pybaseball's FangraphsPitchingStats enum (added in v2.2.7).
    This is the slow legacy-leaderboard scrape — expect it to take a while.
    """
    return pb.pitching_stats(year, qual=0, stat_columns="ALL")


# ---------------------------------------------------------------------------
# 5. Arm angle (tunneling feature — actual release angle, not inferred)
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
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode("utf-8")
    if not text.strip() or text.strip().startswith("<!"):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------
# 6. Pitch tempo (rhythm disruption — a real deception lever, not just pace)
# ---------------------------------------------------------------------------
def pull_pitch_tempo(year: int, min_pitches="1") -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-tempo"
        f"?type=Pit&season_start={year}&season_end={year}"
        f"&n={min_pitches}&game_type=Regular&split=no&with_team_only=1&csv=true"
    )
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode("utf-8")
    if not text.strip() or text.strip().startswith("<!"):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_pitches = []
    all_rates = []
    all_swing_timing = []
    all_stuff = []
    all_arm_angle = []
    all_tempo = []

    for year in YEARS:
        start, end = SEASON_RANGES[year]
        print(f"--- {year} ---")

        print("pulling raw pitch-level statcast...")
        pitches = pull_statcast_pitches(start, end)
        pitches = add_vaa_haa(pitches)
        pitches = add_movement_inches(pitches)
        pitches["season"] = year
        all_pitches.append(pitches)

        print("computing outcome rates...")
        rates = compute_outcome_rates(pitches)
        rates["season"] = year
        all_rates.append(rates)

        print("pulling swing timing leaderboard...")
        try:
            timing = pull_swing_timing(year)
            timing["season"] = year
            all_swing_timing.append(timing)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(1)

        print("pulling Stuff+/Location+/PitchingBot from FanGraphs...")
        try:
            stuff = pull_stuff_plus(year)
            stuff["season"] = year
            all_stuff.append(stuff)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(1)

        print("pulling arm angle leaderboard...")
        try:
            arm_angle = pull_arm_angle(year)
            arm_angle["season"] = year
            all_arm_angle.append(arm_angle)
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(1)

        print("pulling pitch tempo leaderboard...")
        try:
            tempo = pull_pitch_tempo(year)
            tempo["season"] = year
            all_tempo.append(tempo)
        except Exception as e:
            print(f"  FAILED: {e}")

    outputs = {
        "raw/statcast_pitch_level_2025_2026.csv": all_pitches,
        "raw/outcome_rates_by_pitcher_pitchtype_2025_2026.csv": all_rates,
        "raw/swing_timing_by_pitcher_2025_2026.csv": all_swing_timing,
        "stuff_plus_pitchingbot_2025_2026.csv": all_stuff,
        "raw/arm_angle_by_pitcher_2025_2026.csv": all_arm_angle,
        "raw/pitch_tempo_by_pitcher_2025_2026.csv": all_tempo,
    }

    Path("raw").mkdir(exist_ok=True)
    written, skipped = [], []
    for filename, frames in outputs.items():
        if not frames:
            skipped.append(filename)
            continue
        pd.concat(frames, ignore_index=True).to_csv(filename, index=False)
        written.append(filename)

    print(f"Done. {len(written)}/{len(outputs)} CSVs written: {written}")
    if skipped:
        print(f"Skipped (no data — check FAILED lines above): {skipped}")
