"""
Pitcher Deception Project — Merge
===================================
Joins the raw pulls from data_pull.py into two analysis-ready tables:

  1. pitcher_pitchtype_season.csv   (grain: pitcher x pitch_type x season)
     - outcome rates (whiff/chase/strike/gb) from data_pull.py
     - pitch-characteristic means/stds computed here from the raw pitch-level file
       (velo, spin, IVB/HB, VAA/HAA, extension, release point, arm angle)
     - sequencing features computed here (velocity gap from previous pitch in the
       same at-bat, same-pitch-type-as-previous rate)
     - pitcher-season-level covariates broadcast onto every pitch-type row
       (arm angle leaderboard, pitch tempo, pitch-mix entropy)
     - Stuff+/Location+/PitchingBot, if a FanGraphs export has been dropped in
       (see merge_fangraphs_stuff() below) — optional, skipped if absent.
     - games, games_started, innings_pitched, if a second FanGraphs export has
       been dropped in (see merge_fangraphs_standard() below) — optional,
       skipped if absent. export_site_stats.pitcher_roles() prefers the real
       games-started share this gives over its own pitches-per-appearance proxy.

  2. pitcher_season_covariates.csv  (grain: pitcher x season)
     - the same pitcher-season covariates, standalone, for pitcher-level modeling

Run after data_pull.py has produced its CSVs in this same directory.
"""

import glob
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PITCH_LEVEL_FILE = "raw/statcast_pitch_level_2025_2026.csv"


# ---------------------------------------------------------------------------
# 1. Pitch-characteristic aggregates (mean + std) by pitcher x pitch_type x season
# ---------------------------------------------------------------------------
CHAR_COLS = [
    "release_speed", "release_spin_rate", "ivb_in", "hb_in", "vaa", "haa",
    "release_extension", "release_pos_x", "release_pos_z", "arm_angle",
    "effective_speed",
]
AGG_USECOLS = ["pitcher", "pitch_type", "season"] + CHAR_COLS
CHUNKSIZE = 500_000


def compute_pitch_characteristics(path: str) -> pd.DataFrame:
    """Chunked mean/std aggregation — avoids loading the full multi-hundred-MB
    pitch-level file (100+ columns) into memory at once."""
    sums = {}
    sumsqs = {}
    counts = {}

    for chunk in pd.read_csv(path, usecols=AGG_USECOLS, chunksize=CHUNKSIZE, low_memory=False):
        grouped = chunk.groupby(["pitcher", "pitch_type", "season"])
        chunk_sum = grouped[CHAR_COLS].sum(min_count=1)
        chunk_sumsq = grouped[CHAR_COLS].apply(lambda g: (g ** 2).sum(min_count=1))
        chunk_count = grouped[CHAR_COLS].count()

        for key, row in chunk_sum.iterrows():
            sums[key] = sums.get(key, pd.Series(0.0, index=CHAR_COLS)).add(row, fill_value=0)
        for key, row in chunk_sumsq.iterrows():
            sumsqs[key] = sumsqs.get(key, pd.Series(0.0, index=CHAR_COLS)).add(row, fill_value=0)
        for key, row in chunk_count.iterrows():
            counts[key] = counts.get(key, pd.Series(0.0, index=CHAR_COLS)).add(row, fill_value=0)

    records = []
    for key in sums:
        pitcher, pitch_type, season = key
        rec = {"pitcher": pitcher, "pitch_type": pitch_type, "season": season}
        for col in CHAR_COLS:
            n = counts[key][col]
            if n > 0:
                mean = sums[key][col] / n
                var = max(sumsqs[key][col] / n - mean ** 2, 0.0)
                rec[f"{col}_mean"] = mean
                rec[f"{col}_std"] = np.sqrt(var)
            else:
                rec[f"{col}_mean"] = np.nan
                rec[f"{col}_std"] = np.nan
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Sequencing features: velocity gap + repeat rate vs. the previous pitch
#    in the same at-bat (classic tunneling/deception signal)
# ---------------------------------------------------------------------------
SEQ_USECOLS = ["game_pk", "pitcher", "at_bat_number", "pitch_number", "pitch_type",
               "release_speed", "season"]


def compute_sequencing_features(path: str) -> pd.DataFrame:
    seq_chunks = []
    for chunk in pd.read_csv(path, usecols=SEQ_USECOLS, chunksize=CHUNKSIZE, low_memory=False):
        seq_chunks.append(chunk)
    df = pd.concat(seq_chunks, ignore_index=True)
    df = df.sort_values(["pitcher", "game_pk", "at_bat_number", "pitch_number"])

    grp = df.groupby(["pitcher", "game_pk", "at_bat_number"])
    df["prev_pitch_type"] = grp["pitch_type"].shift(1)
    df["prev_release_speed"] = grp["release_speed"].shift(1)

    df["has_prev"] = df["prev_pitch_type"].notna()
    df["velocity_gap"] = (df["release_speed"] - df["prev_release_speed"]).abs()
    df["is_repeat"] = df["pitch_type"] == df["prev_pitch_type"]

    sequenced = df[df["has_prev"]]
    agg = sequenced.groupby(["pitcher", "pitch_type", "season"]).agg(
        avg_velocity_gap_from_prev=("velocity_gap", "mean"),
        repeat_pct=("is_repeat", "mean"),
        n_sequenced=("is_repeat", "size"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# 3. Pitch-mix entropy by pitcher x season (unpredictability of pitch selection —
#    a real deception lever independent of any single pitch's characteristics)
# ---------------------------------------------------------------------------
def compute_pitch_mix_entropy(outcome_rates: pd.DataFrame) -> pd.DataFrame:
    def entropy(group):
        p = group["pitches"] / group["pitches"].sum()
        p = p[p > 0]
        return -(p * np.log2(p)).sum()

    ent = outcome_rates.groupby(["pitcher", "season"]).apply(entropy, include_groups=False)
    return ent.rename("pitch_mix_entropy").reset_index()


# ---------------------------------------------------------------------------
# 4. Loaders for the pitcher-season-level leaderboard pulls
# ---------------------------------------------------------------------------
def load_arm_angle_leaderboard() -> pd.DataFrame:
    df = pd.read_csv("raw/arm_angle_by_pitcher_2025_2026.csv")
    keep = ["pitcher", "season", "ball_angle"]
    return df[keep].rename(columns={"ball_angle": "arm_angle_szn_avg"})


def load_tempo() -> pd.DataFrame:
    df = pd.read_csv("raw/pitch_tempo_by_pitcher_2025_2026.csv")
    df = df.rename(columns={"entity_id": "pitcher"})
    # The pull used split=no, so the file's second pace column is a positional duplicate of this one.
    return df[["pitcher", "season", "median_seconds_empty"]].rename(columns={"median_seconds_empty": "tempo_bases_empty_sec"})


# ---------------------------------------------------------------------------
# 5. Optional FanGraphs Stuff+/Location+/PitchingBot merge (manual export)
# ---------------------------------------------------------------------------
# FanGraphs' leaderboard is behind a live Cloudflare human-verification check
# (a "Verify you are human" Turnstile challenge) that pybaseball's scraper and
# any other script gets blocked by — not just a missing header. Rather than
# script around bot detection, this expects a table exported/pasted by hand
# from the FanGraphs site (pitch-type view with Stuff+/Location+/Pitching+
# columns showing) saved as fangraphs_stuff_*.tsv (tab-separated) or
# fangraphs_stuff_*.csv in raw/, with at least these columns:
#   season, name, stf_fa, stf_si, stf_fc, stf_fs, stf_sl, stf_cu, stf_ch,
#   stf_kc, stf_fo, stuff_plus, location_plus, pitching_plus
#
# FanGraphs exports identify players by name (or FanGraphs' own IDfg), not
# the MLBAM id Statcast uses, and FanGraphs itself is unreachable for an
# automated id crosswalk lookup — so instead of relying on any external id
# service, this builds its own name -> MLBAM-id crosswalk directly from our
# own Statcast pull (pitcher, player_name), which is exact and already local.
#
# Some names belong to two pitchers (Luis García, Jacob Webb, Yunior Marte in the 2025-26 pull). A name
# alone cannot tell them apart, and a plain name -> id dictionary silently keeps the last one it saw, so
# one pitcher received the other's Stuff+ (and, in the standard export, the other's games and starts).
# Those rows are now resolved by the team on the FanGraphs row against the teams each pitcher threw
# for in that season, and left unmatched when that cannot decide.
FANGRAPHS_TEAM_TO_STATCAST = {"ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH"}
FANGRAPHS_PITCH_TYPE_MAP = {
    "stf_fa": "FF", "stf_si": "SI", "stf_fc": "FC", "stf_fs": "FS",
    "stf_sl": "SL", "stf_cu": "CU", "stf_ch": "CH", "stf_kc": "KC", "stf_fo": "FO",
}


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace(".", "").replace("-", " ").replace("'", "")
    return " ".join(s.split())


def statcast_team(fangraphs_team: str) -> str:
    """FanGraphs team abbreviation as Statcast writes it (SFG is SF, CHW is CWS, and so on)."""
    return FANGRAPHS_TEAM_TO_STATCAST.get(fangraphs_team, fangraphs_team)


def build_name_crosswalk(path: str) -> dict[str, list[int]]:
    """normalized 'first last' -> every MLBAM pitcher id with that name, from our own pitch-level pull."""
    ids: dict[str, set[int]] = {}
    for chunk in pd.read_csv(path, usecols=["pitcher", "player_name"], chunksize=CHUNKSIZE):
        # usecols preserves the FILE's column order (player_name precedes
        # pitcher there), not the order listed above — select explicitly
        # rather than unpacking itertuples() positionally.
        for pid, name in chunk[["pitcher", "player_name"]].drop_duplicates().itertuples(index=False):
            last, first = [p.strip() for p in name.split(",", 1)]
            ids.setdefault(normalize_name(f"{first} {last}"), set()).add(int(pid))
    return {name: sorted(pids) for name, pids in ids.items()}


def pitcher_season_teams(path: str) -> pd.DataFrame:
    """Every team each pitcher threw for in the regular season, one row per pitcher and season. The
    pitching team is the home team in the top of an inning and the away team in the bottom."""
    parts = []
    for chunk in pd.read_csv(path, usecols=["pitcher", "season", "game_type", "inning_topbot", "home_team", "away_team"],
                             chunksize=CHUNKSIZE):
        chunk = chunk[chunk["game_type"] == "R"]
        chunk = chunk.assign(team=np.where(chunk["inning_topbot"] == "Top", chunk["home_team"], chunk["away_team"]))
        parts.append(chunk[["pitcher", "season", "team"]].drop_duplicates())
    teams = pd.concat(parts).drop_duplicates()
    return teams.groupby(["pitcher", "season"])["team"].agg(frozenset).rename("teams").reset_index()


def resolve_fangraphs_pitchers(fg: pd.DataFrame, crosswalk: dict[str, list[int]], teams: pd.DataFrame) -> pd.Series:
    """MLBAM id for each FanGraphs row (needs name, season, team), NaN where there is none.

    A name with one pitcher resolves directly. A name shared by several resolves by team: a single-team
    row goes to the pitcher who threw for that team that season, and a "3 Tms" row to the pitcher who threw
    for three. Anything that does not pick out exactly one pitcher is left unmatched, not guessed."""
    season_teams = {(r.pitcher, r.season): r.teams for r in teams.itertuples()}
    resolved = []
    for name, season, team in zip(fg["name"], fg["season"], fg["team"]):
        candidates = crosswalk.get(normalize_name(name), [])
        if len(candidates) > 1:
            label = str(team)
            if label.endswith(" Tms"):
                wanted = int(label.split()[0])
                candidates = [c for c in candidates if len(season_teams.get((c, season), ())) == wanted]
            else:
                candidates = [c for c in candidates if statcast_team(label) in season_teams.get((c, season), ())]
        resolved.append(candidates[0] if len(candidates) == 1 else np.nan)
    return pd.Series(resolved, index=fg.index, dtype="float")


def load_fangraphs_files(prefix: str) -> pd.DataFrame | None:
    """Every fangraphs_{prefix}_*.tsv or .csv file in raw/, concatenated. None if there are none."""
    files = glob.glob(f"raw/{prefix}_*.tsv") + glob.glob(f"raw/{prefix}_*.csv")
    if not files:
        return None
    frames = [pd.read_csv(f, sep="\t" if f.endswith(".tsv") else ",") for f in files]
    return pd.concat(frames, ignore_index=True)


def merge_fangraphs_stuff(pitcher_season_covariates: pd.DataFrame):
    """Returns (pitch_type_long_df_or_None, updated pitcher_season_covariates).
    The pitch-type frame is merged into the pitcher x pitch_type x season table
    by the caller; the pitcher-season overall Stuff+/Location+/Pitching+ score
    is folded straight into pitcher_season_covariates like every other
    pitcher-season covariate."""
    fg = load_fangraphs_files("fangraphs_stuff")
    if fg is None:
        print("No fangraphs_stuff_*.tsv/csv file found — skipping Stuff+ merge. "
              "See the comment above merge_fangraphs_stuff() for the expected format.")
        return None, pitcher_season_covariates

    if "name" not in fg.columns or "season" not in fg.columns:
        raise ValueError(
            f"Expected 'name' and 'season' columns in the FanGraphs export, got {list(fg.columns)}"
        )

    crosswalk = build_name_crosswalk(PITCH_LEVEL_FILE)
    fg["pitcher"] = resolve_fangraphs_pitchers(fg, crosswalk, pitcher_season_teams(PITCH_LEVEL_FILE))

    unmatched = fg["pitcher"].isna().sum()
    if unmatched:
        sample = fg.loc[fg["pitcher"].isna(), "name"].unique()[:10]
        print(f"Warning: {unmatched}/{len(fg)} FanGraphs rows had no match in our own "
              f"Statcast pull and will be dropped from the Stuff+ merge. "
              f"Sample unmatched names: {list(sample)}")
    shared = fg["name"].map(normalize_name).map(lambda n: len(crosswalk.get(n, [])) > 1)
    print(f"{int(shared.sum())} Stuff+ rows carry a name shared by two pitchers; "
          f"{int((shared & fg['pitcher'].notna()).sum())} resolved by team.")
    fg = fg.dropna(subset=["pitcher"]).copy()
    fg["pitcher"] = fg["pitcher"].astype(int)
    # FanGraphs' own export already collapses a mid-season trade into one
    # "2 Tms"/"3 Tms" row, so a pitcher should appear at most once per season.
    fg = fg.drop_duplicates(subset=["pitcher", "season"], keep="first")

    pitch_type_cols = [c for c in FANGRAPHS_PITCH_TYPE_MAP if c in fg.columns]
    pitch_type_long = fg.melt(
        id_vars=["pitcher", "season"], value_vars=pitch_type_cols,
        var_name="_fg_col", value_name="stuff_plus_by_pitch_type",
    ).dropna(subset=["stuff_plus_by_pitch_type"])
    pitch_type_long["pitch_type"] = pitch_type_long["_fg_col"].map(FANGRAPHS_PITCH_TYPE_MAP)
    pitch_type_long = pitch_type_long[["pitcher", "season", "pitch_type", "stuff_plus_by_pitch_type"]]

    overall_cols = [c for c in ("stuff_plus", "location_plus", "pitching_plus") if c in fg.columns]
    overall = fg[["pitcher", "season"] + overall_cols]
    pitcher_season_covariates = pitcher_season_covariates.merge(
        overall, on=["pitcher", "season"], how="outer"
    )

    return pitch_type_long, pitcher_season_covariates


# ---------------------------------------------------------------------------
# 6. Optional games/starts/innings merge (manual export, real role instead of a proxy)
# ---------------------------------------------------------------------------
# Same reason as the Stuff+ export above: FanGraphs blocks scripted access. Expects a standard
# pitching leaderboard exported/pasted by hand as fangraphs_standard_*.tsv or .csv in raw/, with at
# least Season, Name, G, GS, IP. Unlike the Stuff+ export, this one does NOT collapse a mid-season
# trade into one row — a traded pitcher gets one row per team with no combined total — so rows are
# summed by (pitcher, season) rather than deduplicated, which gives the right season total for a
# real trade. Each row carries its own team, so a name shared by two pitchers is resolved row by row
# the same way as in the Stuff+ merge, and each pitcher's rows are summed separately.
def merge_fangraphs_standard(pitcher_season_covariates: pd.DataFrame) -> pd.DataFrame:
    """Adds games, games_started, innings_pitched to pitcher_season_covariates, for a real
    games-started-share role label instead of export_site_stats.py's pitches-per-appearance proxy."""
    fg = load_fangraphs_files("fangraphs_standard")
    if fg is None:
        print("No fangraphs_standard_*.tsv/csv file found — skipping the games/starts/innings merge. "
              "See the comment above merge_fangraphs_standard() for the expected format.")
        return pitcher_season_covariates

    fg = fg.rename(columns={"Season": "season", "Name": "name", "G": "games", "GS": "games_started", "IP": "innings_pitched",
                            "Team": "team"})
    missing = {"season", "name", "games", "games_started", "innings_pitched"} - set(fg.columns)
    if missing:
        raise ValueError(f"Expected Season, Name, G, GS, IP in the FanGraphs standard export, missing {missing}")

    crosswalk = build_name_crosswalk(PITCH_LEVEL_FILE)
    fg["pitcher"] = resolve_fangraphs_pitchers(fg, crosswalk, pitcher_season_teams(PITCH_LEVEL_FILE))
    unmatched = fg["pitcher"].isna().sum()
    if unmatched:
        sample = fg.loc[fg["pitcher"].isna(), "name"].unique()[:10]
        print(f"Warning: {unmatched}/{len(fg)} FanGraphs standard rows had no match in our own "
              f"Statcast pull and will be dropped. Sample unmatched names: {list(sample)}")
    fg = fg.dropna(subset=["pitcher"]).copy()
    fg["pitcher"] = fg["pitcher"].astype(int)

    totals = fg.groupby(["pitcher", "season"])[["games", "games_started", "innings_pitched"]].sum().reset_index()
    return pitcher_season_covariates.merge(totals, on=["pitcher", "season"], how="outer")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("loading outcome rates...")
    outcome_rates = pd.read_csv("raw/outcome_rates_by_pitcher_pitchtype_2025_2026.csv")

    print("computing pitch characteristics (chunked pass over pitch-level file)...")
    pitch_chars = compute_pitch_characteristics(PITCH_LEVEL_FILE)

    print("computing sequencing features...")
    sequencing = compute_sequencing_features(PITCH_LEVEL_FILE)

    print("computing pitch-mix entropy...")
    entropy = compute_pitch_mix_entropy(outcome_rates)

    print("loading pitcher-season leaderboards (arm angle, tempo)...")
    arm_angle = load_arm_angle_leaderboard()
    tempo = load_tempo()

    print("merging pitcher x pitch_type x season table...")
    table = outcome_rates.merge(pitch_chars, on=["pitcher", "pitch_type", "season"], how="left")
    table = table.merge(sequencing, on=["pitcher", "pitch_type", "season"], how="left")

    pitcher_season_covariates = (
        entropy
        .merge(arm_angle, on=["pitcher", "season"], how="outer")
        .merge(tempo, on=["pitcher", "season"], how="outer")
    )

    print("attempting FanGraphs Stuff+/Location+/PitchingBot merge (optional)...")
    fg_pitch_type_long, pitcher_season_covariates = merge_fangraphs_stuff(pitcher_season_covariates)

    print("attempting FanGraphs games/starts/innings merge (optional)...")
    pitcher_season_covariates = merge_fangraphs_standard(pitcher_season_covariates)

    table = table.merge(pitcher_season_covariates, on=["pitcher", "season"], how="left")
    if fg_pitch_type_long is not None:
        table = table.merge(fg_pitch_type_long, on=["pitcher", "season", "pitch_type"], how="left")

    Path("output").mkdir(exist_ok=True)
    table.to_csv("output/pitcher_pitchtype_season.csv", index=False)
    pitcher_season_covariates.to_csv("output/pitcher_season_covariates.csv", index=False)

    print(f"Done. pitcher_pitchtype_season.csv: {table.shape[0]} rows, {table.shape[1]} cols")
    print(f"      pitcher_season_covariates.csv: {pitcher_season_covariates.shape[0]} rows, "
          f"{pitcher_season_covariates.shape[1]} cols")
