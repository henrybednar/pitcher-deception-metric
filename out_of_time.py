"""
Pitcher Deception Project: out-of-time replication
==================================================
Every design decision in the project (which components are members, their weights, the forecast gate)
was made on 2025 and 2026, and the stability figures were then reported on the same pairs. This script
runs the unmodified pipeline on a different pair, real 2024 and 2025, and reports the same statistics
side by side, so a decision made on one pair can be checked on the other.

How: a sandbox directory gets a copy of the code, a raw pitch file where real 2024 is labelled 2025 and
real 2025 is labelled 2026 (the code reads those two seasons and nothing else), and the matching
FanGraphs Stuff+ rows. The fit, scoring and forecast steps then run inside it, and compare() reads
both sets of outputs. Nothing in the project's own raw/ or output/ folders is touched.

    python out_of_time.py                run everything (pull 2024, build, run, compare)
    python out_of_time.py --compare      only compare, when the sandbox has already run
    python out_of_time.py --sandbox DIR  use another sandbox location (default sandbox_out_of_time)

The pull needs the network (Baseball Savant) and the run takes about as long as the pipeline's model
steps (roughly 20 minutes). Bat tracking (the timing and whiff miss inputs) is present in the 2024
pitch-level file. Arm angle and tempo leaderboards are not needed, because the driver steps are skipped.
"""

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
DEFAULT_SANDBOX = REPO / "sandbox_out_of_time"
HOLDOUT_YEAR = 2024
RAW_FILE = "raw/statcast_pitch_level_2025_2026.csv"
FANGRAPHS_STUFF_FILE = "raw/fangraphs_stuff_manual.tsv"
STEPS = ["build_pitch_table", "fit_full_model", "fit_swing_alignment", "fit_called_strike", "reliability_and_ci",
         "predictive_validity"]


def shift_seasons(frame: pd.DataFrame, by: int = 1) -> pd.DataFrame:
    """Same rows with season moved by `by` (real 2024 becomes the code's 2025)."""
    return frame.assign(season=frame["season"] + by)


def holdout_rows(raw_chunks, real_season: int, labelled_as: int):
    """The rows of one real season from an iterable of raw chunks, labelled as another season."""
    for chunk in raw_chunks:
        chunk = chunk[chunk["season"] == real_season]
        if len(chunk):
            yield chunk.assign(season=labelled_as)


def weighted_composite(ps: pd.DataFrame, labels: list[str], reliabilities: dict[str, float], minimum: float = 0.2) -> pd.Series:
    """The composite for any set of members, on the 100/10 scale of the qualified population, with the pipeline's
    weights (a member's reliability times the pitcher's sample over that member's median sample) and its
    rule that at least two members are needed. Members under `minimum` reliability get no weight."""
    z, w = [], []
    for label in labels:
        centred = (ps[f"{label}_index"] - 100) / 10
        median_n = ps.loc[ps[f"{label}_index"].notna(), f"{label}_n"].median()
        reliability = reliabilities[label] if reliabilities[label] >= minimum else 0.0
        z.append(centred.fillna(0).to_numpy())
        w.append(np.where(centred.notna(), reliability * ps[f"{label}_n"] / median_n, 0.0))
    z, w = np.array(z), np.array(w)
    total = w.sum(axis=0)
    raw = np.where(((w > 0).sum(axis=0) >= 2) & (total > 0), (z * w).sum(axis=0) / np.where(total == 0, np.nan, total), np.nan)
    qualified = ps["qualified"].to_numpy() & ~np.isnan(raw)
    return pd.Series(100 + 10 * (raw - raw[qualified].mean()) / raw[qualified].std(ddof=1), index=ps.index)


def qualified_pair_yoy(ps: pd.DataFrame, score: pd.Series, n_boot: int = 400, seed: int = 5) -> tuple[float, float, float, int]:
    """Year-over-year correlation of `score` over pitchers qualified in both seasons, with a 95% interval
    from resampling pitchers."""
    d = ps.assign(score=score)
    d = d[d["qualified"] & d["score"].notna()]
    both = d.pivot_table(index="pitcher", columns="season", values="score").dropna()
    first, second = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    rng = np.random.default_rng(seed)
    draws = [np.corrcoef(first[i], second[i])[0, 1] for i in (rng.integers(0, len(first), len(first)) for _ in range(n_boot))]
    return float(np.corrcoef(first, second)[0, 1]), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), int(len(first))


def pull_holdout(path: Path) -> None:
    """Pull real 2024 from Baseball Savant, labelled 2025. data_pull is imported only here: importing it
    clears the pybaseball cache directory, as every pull does."""
    import data_pull as dp

    pitches = dp.add_movement_inches(dp.add_vaa_haa(dp.pull_statcast_pitches(f"{HOLDOUT_YEAR}-03-20", f"{HOLDOUT_YEAR}-11-01")))
    pitches["season"] = 2025
    pitches.to_csv(path, index=False)
    print(f"pulled {len(pitches):,} pitches for {HOLDOUT_YEAR}", flush=True)


def build_sandbox(sandbox: Path) -> None:
    for sub in ("raw", "output"):
        (sandbox / sub).mkdir(parents=True, exist_ok=True)
    for source in glob.glob(str(REPO / "*.py")):
        shutil.copy(source, sandbox)
    pull = sandbox / f"pull_{HOLDOUT_YEAR}.csv"
    if not pull.exists():
        pull_holdout(pull)
    parts = [pd.read_csv(pull, low_memory=False)]
    parts += list(holdout_rows(pd.read_csv(REPO / RAW_FILE, chunksize=300_000, low_memory=False), 2025, 2026))
    pd.concat(parts, ignore_index=True).to_csv(sandbox / RAW_FILE, index=False)
    fg = pd.read_csv(REPO / FANGRAPHS_STUFF_FILE, sep="\t")
    shift_seasons(fg[fg["season"].isin([HOLDOUT_YEAR, HOLDOUT_YEAR + 1])]).to_csv(sandbox / FANGRAPHS_STUFF_FILE, sep="\t", index=False)
    covariates = (
        "import pandas as pd, merge_data\n"
        "cov = pd.DataFrame({'pitcher': pd.Series(dtype='int64'), 'season': pd.Series(dtype='int64')})\n"
        "_, cov = merge_data.merge_fangraphs_stuff(cov)\n"
        "cov.to_csv('output/pitcher_season_covariates.csv', index=False)\n")
    subprocess.run([sys.executable, "-c", covariates], cwd=sandbox, check=True)


def run_steps(sandbox: Path) -> None:
    for step in STEPS:
        print(f"=== {step}", flush=True)
        subprocess.run([sys.executable, "-u", f"{step}.py"], cwd=sandbox, check=True)


def compare(sandbox: Path) -> dict:
    """Print the main (real 2025 to 2026) and out-of-time (real 2024 to 2025) statistics side by side."""
    import export_site_stats as ess
    import reliability_and_ci as rc

    rng = np.random.default_rng(5)
    main_ps = pd.read_csv(REPO / "output/pitcher_season.csv")
    oot_ps = pd.read_csv(sandbox / "output/pitcher_season.csv")
    main_rel = pd.read_csv(REPO / "output/reliability_report.csv").set_index("label")
    oot_rel = pd.read_csv(sandbox / "output/reliability_report.csv").set_index("label")
    main_pv = pd.read_csv(REPO / "output/predictive_validity_report.csv").set_index("label")
    oot_pv = pd.read_csv(sandbox / "output/predictive_validity_report.csv").set_index("label")
    members = rc.COMPOSITE_OUTCOMES
    summary = {}
    print("main = real 2025 -> 2026; out of time = real 2024 -> 2025\n")

    print("reliability (split-half, Spearman-Brown)")
    rows = [(l, main_rel.loc[l, "r_full_spearman_brown"], oot_rel.loc[l, "r_full_spearman_brown"])
            for l in members + ["gb", "calledstrike", "align", "deception_plus"]]
    print(pd.DataFrame(rows, columns=["label", "main", "oot"]).round(3).to_string(index=False))
    summary["reliability"] = {l: {"main": float(a), "oot": float(b)} for l, a, b in rows}

    print("\nyear-over-year correlation by component")
    rows = [(l, ess.component_yoy(main_ps, l), ess.component_yoy(oot_ps, l)) for l in members + ["gb", "calledstrike", "align"]]
    print(pd.DataFrame(rows, columns=["component", "main", "oot"]).round(3).to_string(index=False))
    for name, ps in (("main", main_ps), ("oot", oot_ps)):
        scored = ps.dropna(subset=["deception_plus"])
        joined = scored[scored["season"] == 2025].merge(scored[scored["season"] == 2026], on="pitcher")
        q = ess.year_over_year_summary(ps)
        print(f"  composite {name}: all scored r={joined.deception_plus_x.corr(joined.deception_plus_y):.3f} (n={len(joined)}); "
              f"qualified pairs r={q['r']:.3f} [{q['lo']:.3f}, {q['hi']:.3f}] n={q['n']}; higher-volume half {q['high_volume_r']:.3f}, lower {q['low_volume_r']:.3f}")
        summary[f"composite_yoy_{name}"] = q

    print("\nforecast test: gain in cross-validated R2 from the prior season's score, and its p-value")
    rows = [(l, main_pv.loc[l, "cv_delta_r2"], main_pv.loc[l, "f_pvalue"], oot_pv.loc[l, "cv_delta_r2"], oot_pv.loc[l, "f_pvalue"])
            for l in members + ["gb", "calledstrike", "align", "composite_to_whiff"]]
    print(pd.DataFrame(rows, columns=["label", "main dR2", "main p", "oot dR2", "oot p"]).round(4).to_string(index=False))

    print("\ncomposite year-over-year among qualified pairs for other memberships (does the membership decision replicate?)")
    main_weights = {l: float(main_rel.loc[l, "r_full_spearman_brown"]) for l in main_rel.index}
    oot_weights = {l: float(oot_rel.loc[l, "r_full_spearman_brown"]) for l in oot_rel.index}
    variants = {"shipped": members, "+ground ball": members + ["gb"], "+called strike": members + ["calledstrike"],
                "+alignment": members + ["align"], "-weak contact": [m for m in members if m != "weak"],
                "-timing": [m for m in members if m != "timing"], "-whiff miss": [m for m in members if m != "whiffmiss"],
                "-chase": [m for m in members if m != "chase"], "-whiff": [m for m in members if m != "whiff"]}
    summary["memberships"] = {}
    for name, labels in variants.items():
        a = qualified_pair_yoy(main_ps, weighted_composite(main_ps, labels, main_weights))
        b = qualified_pair_yoy(oot_ps, weighted_composite(oot_ps, labels, oot_weights))
        print(f"  {name:15s} main {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]   out of time {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] n={b[3]}")
        summary["memberships"][name] = {"main": a[0], "oot": b[0]}

    print("\nStuff+ link")
    for name, ps in (("main", main_ps), ("oot", oot_ps)):
        s = ess.stuff_plus_correlations(ps)
        lo, hi = ess.stuff_plus_correlation_interval(ps, n_boot=500)
        print(f"  {name}: composite r={s['composite']:.3f} [{lo:.3f}, {hi:.3f}] n={s['n']}; members {({k: round(v, 2) for k, v in s['members'].items()})}")
        summary[f"stuff_plus_{name}"] = {"r": s["composite"], "lo": lo, "hi": hi}

    print("\nthe same real season (2025) scored by both runs")
    a = main_ps[(main_ps.season == 2025) & main_ps.qualified][["pitcher", "deception_plus"]]
    b = oot_ps[(oot_ps.season == 2026) & oot_ps.qualified][["pitcher", "deception_plus"]]
    j = a.merge(b, on="pitcher", suffixes=("_main", "_oot")).dropna()
    d = j.deception_plus_main - j.deception_plus_oot
    print(f"  {len(j)} pitchers: correlation {j.deception_plus_main.corr(j.deception_plus_oot):.3f}, SD of the difference {d.std():.2f} points")
    summary["cross_run_2025"] = {"n": int(len(j)), "corr": float(j.deception_plus_main.corr(j.deception_plus_oot)), "sd_diff": float(d.std())}

    print("\nthree seasons (real 2024 from the sandbox, real 2025 and 2026 from the main run), qualified in all")
    t = pd.concat([oot_ps[(oot_ps.season == 2025) & oot_ps.qualified].set_index("pitcher")["deception_plus"].rename("y24"),
                   main_ps[(main_ps.season == 2025) & main_ps.qualified].set_index("pitcher")["deception_plus"].rename("y25"),
                   main_ps[(main_ps.season == 2026) & main_ps.qualified].set_index("pitcher")["deception_plus"].rename("y26")], axis=1).dropna()
    avg = t[["y24", "y25"]].mean(axis=1)
    print(f"  {len(t)} pitchers: predicting 2026, r with 2025 {t.y26.corr(t.y25):.3f}, with 2024 {t.y26.corr(t.y24):.3f}, with the 2024-25 average {t.y26.corr(avg):.3f}")
    summary["three_seasons"] = {"n": int(len(t)), "r_2025": float(t.y26.corr(t.y25)), "r_2024": float(t.y26.corr(t.y24)), "r_average": float(t.y26.corr(avg))}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sandbox", type=Path, default=DEFAULT_SANDBOX)
    parser.add_argument("--compare", action="store_true", help="only compare; the sandbox has already run")
    args = parser.parse_args()
    if not args.compare:
        build_sandbox(args.sandbox)
        run_steps(args.sandbox)
    summary = compare(args.sandbox)
    (REPO / "output").mkdir(exist_ok=True)
    with open(REPO / "output/out_of_time.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
