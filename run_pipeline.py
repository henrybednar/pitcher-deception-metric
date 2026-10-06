"""
Pitcher Deception Project: run the whole pipeline in order
==========================================================
    python run_pipeline.py              run every step from the raw files
    python run_pipeline.py --list       show the steps and what each one writes
    python run_pipeline.py --from fit_full_model
    python run_pipeline.py --only build_pages
    python run_pipeline.py --pull       run data_pull.py first (network, slow)

Each step runs as its own process with this directory as the working
directory. Output is echoed and saved to logs/<step>.log. The run stops at the
first step that fails or does not write the files it should.

The raw files (see RAW_INPUTS) are not rebuilt unless you pass --pull. A pull
fetches whatever Savant has today, so the data cutoff and every downstream
number move with it.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
FILE_TIME_SLACK_SECONDS = 2.0     # file systems stamp modification times a little coarsely

RAW_INPUTS = [
    "raw/statcast_pitch_level_2024_2026.csv",
    "raw/outcome_rates_by_pitcher_pitchtype_2024_2026.csv",
    "raw/swing_timing_by_pitcher_2024_2026.csv",
    "raw/arm_angle_by_pitcher_2024_2026.csv",
    "raw/pitch_tempo_by_pitcher_2024_2026.csv",
    "raw/fangraphs_stuff_manual.tsv",
]


@dataclass(frozen=True)
class Step:
    name: str
    scripts: tuple[str, ...]
    outputs: tuple[str, ...]
    note: str


PULL_STEP = Step("data_pull", ("data_pull.py",), tuple(RAW_INPUTS[:-1]), "download raw Statcast and leaderboard files")

STEPS = [
    Step("merge_data", ("merge_data.py",),
         ("output/pitcher_pitchtype_season.csv", "output/pitcher_season_covariates.csv"),
         "join raw files into pitcher-level covariate tables"),
    Step("driver_features", ("driver_features.py",),
         ("output/driver_features.csv",),
         "season-level candidate driver features"),
    Step("build_pitch_table", ("build_pitch_table.py",),
         ("output/per_pitch_predictions.csv",), "per-pitch features, targets and half assignment"),
    Step("fit_full_model", ("fit_full_model.py",), ("output/per_pitch_predictions.csv",),
         "whiff, chase, ground ball, weak contact: stuff, location, batter, catcher, count, park, platoon"),
    Step("fit_swing_alignment", ("fit_swing_alignment.py",), ("output/per_pitch_predictions.csv",),
         "timing, horizontal alignment and whiff miss distance against each hitter's ideal contact point"),
    Step("fit_timing_direction", ("fit_timing_direction.py",), ("output/per_pitch_predictions.csv",),
         "signed timing model, stuff+location baseline (readout, not scored)"),
    Step("fit_called_strike", ("fit_called_strike.py",), ("output/per_pitch_predictions.csv",),
         "called strike model (validation table only)"),
    Step("reliability_and_ci", ("reliability_and_ci.py",),
         ("output/pitcher_season.csv", "output/reliability_report.csv", "output/half_scores.csv", "output/design_effects.json"),
         "shrinkage, bootstrap intervals, composite, split-half reliability, player names and Stuff+"),
    Step("add_timing_direction", ("add_timing_direction.py",), ("output/pitcher_season.csv", "output/reliability_report.csv"),
         "signed timing diagnostic with intervals"),
    Step("interval_coverage", ("interval_coverage.py",), ("output/interval_coverage.json",),
         "are the shrinkage intervals calibrated? one half of a season's games predicts the other half"),
    Step("driver_analysis", ("driver_analysis.py",), ("output/driver_analysis.json",),
         "what explains each residual? (season level)"),
    Step("predictive_validity", ("predictive_validity.py",), ("output/predictive_validity_report.csv",),
         "does a season's score predict the next season's outcome rate?"),
    Step("projection", ("projection.py",), ("output/projection.csv", "output/projection.json"),
         "projected Deception+ for the next season from the last two (out of sample)"),
    Step("outcome_validation", ("outcome_validation.py",), ("output/outcome_validation.json",),
         "does Deception+ predict next-season strikeout rate, xwOBA and run value beyond Stuff+ and Location+?"),
    Step("membership_check", ("membership_check.py",), ("output/membership_check.json",),
         "does the composite's membership hold? (each member dropped, each candidate added)"),
    Step("sequencing_driver_analysis", ("sequencing_driver_analysis.py",), ("output/sequencing_driver_report.json",),
         "does the previous pitch explain the residual? (pitch level)"),
    Step("pitch_surprise", ("pitch_surprise.py",), ("output/pitch_surprise.json",),
         "does an unpredictable pitch beat its expectation? (pitch and season level)"),
    Step("stuffplus_relationship", ("stuffplus_relationship.py",), ("output/stuffplus_relationship.json",),
         "average result by Stuff+ fifth, and the Stuff+ slope by pitch type and on first pitches only"),
    Step("model_validation", ("model_validation.py",), ("output/model_validation.json",),
         "out-of-fold AUC, log loss, calibration and R2 for every pitch model"),
    Step("pitch_type_scores", ("pitch_type_scores.py",), ("output/pitch_type_scores.csv", "output/pitch_type_scores.json"),
         "whiff and chase by pitch type, with the reliability of each (a pitcher's slider against every other slider)"),
    Step("export_data", ("export_artifact_data.py", "export_site_stats.py", "export_leaderboard_data.py"),
         ("output/artifact_data.json", "output/site_stats.json", "output/leaderboard_data.json"),
         "JSON and stats consumed by the pages"),
    Step("build_pages", ("build_artifact.py", "build_methodology.py", "build_leaderboard.py"),
         ("deception_dashboard.html", "deception_methodology.html", "deception_leaderboard.html"),
         "standalone HTML pages (add --artifact to a build script for the Claude artifact fragment)"),
    Step("check_readme_claims", ("check_readme_claims.py",), ("output/readme_claims_check.txt",),
         "last, so a stale README never blocks the pages: do the README's headline numbers match this run's outputs? (fails when one is stale; fix the README, then --only check_readme_claims)"),
]


def check_raw_inputs() -> list[str]:
    return [name for name in RAW_INPUTS if not (ROOT / name).exists()]


def run_script(script: str, log) -> int:
    proc = subprocess.Popen(
        [sys.executable, "-u", script], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    return proc.wait()


def run_step(step: Step) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    started = time.time()
    print(f"\n=== {step.name}: {step.note}", flush=True)
    with open(LOG_DIR / f"{step.name}.log", "w", encoding="utf-8") as log:
        for script in step.scripts:
            code = run_script(script, log)
            if code != 0:
                raise SystemExit(f"{script} exited with code {code}. See logs/{step.name}.log")
    missing = [o for o in step.outputs if not (ROOT / o).exists()]
    if missing:
        raise SystemExit(f"{step.name} finished but did not write: {missing}")
    # an output left over from an earlier run must not pass for this step's: it has to have been written since the step began
    stale = [o for o in step.outputs if (ROOT / o).stat().st_mtime < started - FILE_TIME_SLACK_SECONDS]
    if stale:
        raise SystemExit(f"{step.name} finished but did not update: {stale}")
    print(f"=== {step.name} done in {(time.time() - started) / 60:.1f} min", flush=True)


def select_steps(args: argparse.Namespace) -> list[Step]:
    names = [s.name for s in STEPS]
    if args.only:
        if args.only not in names:
            raise SystemExit(f"unknown step {args.only!r}. Choices: {names}")
        return [s for s in STEPS if s.name == args.only]
    if args.start:
        if args.start not in names:
            raise SystemExit(f"unknown step {args.start!r}. Choices: {names}")
        return STEPS[names.index(args.start):]
    return list(STEPS)


def main() -> None:
    # player names in the raw data include replacement characters that a cp1252 console cannot print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the steps and exit")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--from", dest="start", help="start at this step")
    where.add_argument("--only", help="run just this step")
    parser.add_argument("--pull", action="store_true", help="run data_pull.py first (network)")
    args = parser.parse_args()

    if args.list:
        for i, step in enumerate(STEPS, 1):
            print(f"{i:2d}. {step.name:28s} {step.note}")
            print(f"    writes: {', '.join(step.outputs)}")
        return

    (ROOT / "raw").mkdir(exist_ok=True)
    (ROOT / "output").mkdir(exist_ok=True)
    steps = select_steps(args)
    if args.pull:
        steps = [PULL_STEP] + steps
    if not args.pull:
        missing = check_raw_inputs()
        if missing:
            raise SystemExit(f"missing raw input files: {missing}. Run with --pull to download them.")

    began = time.time()
    for step in steps:
        run_step(step)
    print(f"\nPipeline finished in {(time.time() - began) / 60:.1f} min.")


if __name__ == "__main__":
    main()
