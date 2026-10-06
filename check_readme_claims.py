"""
Pitcher Deception Project: do the README's headline numbers match the outputs?
===============================================================================
The pages take their figures from site_stats.json, so they cannot drift. The README is written by hand, and
after every rerun some sentence kept an old number (several were found by reading, after the fact). This
script reads each headline sentence, pulls the figures out of it and compares them with the pipeline's own
files. Run it after a pipeline run:

    python check_readme_claims.py

It exits 1 and lists every sentence that disagrees or no longer matches its pattern (reworded
sentences need their pattern updated here, which is the point: a reworded claim gets looked at).
"""

import json
import re
import sys
from pathlib import Path
from typing import Callable, NamedTuple

README = "README.md"
REPORT = "output/readme_claims_check.txt"
COMPONENT_ORDER = ["whiff", "chase", "gb", "weak", "timing", "whiffmiss"]


class Claim(NamedTuple):
    name: str
    pattern: str                                  # one sentence of the README, with a group per figure
    expected: Callable[[dict], list[str]]         # the figures it should hold, as the README prints them


def load_outputs(output_dir: str = "output") -> dict:
    """The output files the claims read, keyed by short names. Missing files are left out."""
    files = {"text": ("site_stats.json", lambda d: d["text"]), "outcome": ("outcome_validation.json", lambda d: d),
             "projection": ("projection.json", lambda d: d), "drivers": ("driver_analysis.json", lambda d: d),
             "membership": ("membership_check.json", lambda d: d), "validation": ("model_validation.json", lambda d: d["outcomes"])}
    out = {}
    for key, (name, pick) in files.items():
        path = Path(output_dir) / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if key == "text" and "text" not in data:
                continue                                  # a site_stats.json without text fields: the claims that need it report themselves
            out[key] = pick(data)
    return out


def gain(g: dict) -> list[str]:
    return [f"{g['gain']:+.3f}", f"{g['lo']:+.3f}", f"{g['hi']:+.3f}"]


def on_field(o: dict) -> list[str]:
    k, x = o["outcomes"]["k_pct"], o["outcomes"]["xwoba"]
    return ([f"{k['r2']['stuff_location']:.3f}", f"{k['r2']['stuff_location_deception']:.3f}"] + gain(k["deception_over_stuff_location"])
            + [f"{x['r2']['stuff_location']:.3f}", f"{x['r2']['stuff_location_deception']:.3f}"] + gain(x["deception_over_stuff_location"]))


def survivors(drivers: dict) -> list[str]:
    """Survivors per outcome in the README's order, with the number of features tested (the "of 14") after the first."""
    counts = [str(sum(f["q"] < 0.05 for f in drivers[c]["features"])) for c in COMPONENT_ORDER]
    return [counts[0], str(len(drivers[COMPONENT_ORDER[0]]["features"]))] + counts[1:]


def projection_coefficients(p: dict) -> list[str]:
    one, two = p["coefficients"]["one_season"], p["coefficients"]["two_seasons"]
    return [f"{one['intercept']:.1f}", f"{one['this']:.2f}", f"{two['intercept']:.1f}", f"{two['this']:.2f}", f"{two['previous']:.2f}"]


def projection_errors(p: dict) -> list[str]:
    b = p["backtest"]
    return [f"{b['rmse_projection']:.1f}", f"{b['rmse_raw_score']:.1f}", f"{b['rmse_league_average']:.1f}"]


def interval(entry: dict) -> list[str]:
    return [f"{abs(entry['diff']):.3f}", f"{entry['diff_lo']:+.3f}", f"{entry['diff_hi']:+.3f}"]


def membership(m: dict) -> list[str]:
    """The README's sentence on adding called strike and alignment and dropping whiff miss and whiff."""
    return interval(m["+calledstrike"]) + interval(m["+align"]) + interval(m["-whiffmiss"]) + interval(m["-whiff"])


def stuff_by_season(text: dict) -> list[str]:
    return re.findall(r"\d\.\d{3}", text["STUFF_R_BY_SEASON"])


def stuff_by_member(text: dict) -> list[str]:
    return [f"{float(x):.2f}" for x in re.findall(r"\.\d{3}", text["STUFF_R_MEMBERS"])]


def metric_range(validation: dict, label: str, metric: str, decimals: int = 2) -> list[str]:
    values = [t[metric] for t in validation[label]["by_pitch_type"]]
    return [f"{min(values):.{decimals}f}", f"{max(values):.{decimals}f}"]


def auc_ranges(v: dict) -> list[str]:
    return sum((metric_range(v, label, "auc") for label in ("whiff", "chase", "weak", "gb", "calledstrike")), [])


def log_losses(v: dict) -> list[str]:
    return sum(([f"{v[label]['overall']['logloss']:.3f}", f"{v[label]['overall']['baseline_logloss']:.3f}"] for label in ("whiff", "chase", "weak", "gb")), [])


def r2_ranges(v: dict) -> list[str]:
    return sum((metric_range(v, label, "r2") for label in ("timing", "whiffmiss", "align")), [])


def stability(o: dict) -> list[str]:
    r = o["stability"]["r"]
    return [f"{r['deception_plus']:.2f}", str(o["stability"]["n"]), f"{r['whiff_rate']:.2f}", f"{r['stuff_plus']:.2f}"]


CLAIMS = [
    Claim("composite reliability and year-over-year",
          r"is (\d\.\d{3}) and year-over-year correlation is (\d\.\d{3}) across every scored pair",
          lambda s: [s["text"]["COMPOSITE_R"], s["text"]["YOY_R"]]),
    Claim("year-over-year among qualified pairs",
          r"and (\d\.\d{3}) among the (\d+) pairs where the pitcher qualified in both seasons",
          lambda s: [s["text"]["YOY_QUAL_R"], s["text"]["YOY_QUAL_N"]]),
    Claim("Stuff+ correlation",
          r"correlates (\d\.\d{3}) with Stuff\+ among qualified pitcher-seasons \(pitcher-clustered 95% interval (\d\.\d{2}) to (\d\.\d{2})",
          lambda s: [s["text"]["STUFF_R"], *s["text"]["STUFF_R_CI"].split(" to ")]),
    Claim("reliever and starter means",
          r"a smaller gap remains: qualified relievers average (\d+\.\d) against (\d+\.\d) for starters",
          lambda s: [s["text"]["RP_MEAN"], s["text"]["SP_MEAN"]]),
    Claim("scores clear of 100",
          r"so only (\d+) of ([\d,]+) qualified scores \((\d+)%\)",
          lambda s: [s["text"]["DP_CLEAR"], s["text"]["N_QUALIFIED"], s["text"]["DP_CLEAR_PCT"].rstrip("%")]),
    Claim("Stuff+ correlation by season",
          r"\((?:pitcher-clustered 95% interval \d\.\d{2} to \d\.\d{2}; )(\d\.\d{3}) in 2024, (\d\.\d{3}) in 2025, (\d\.\d{3}) in 2026\)",
          lambda s: stuff_by_season(s["text"])),
    Claim("Stuff+ correlation by member",
          r"every composite member does too: whiff (\d\.\d{2}), chase (\d\.\d{2}), weak contact (\d\.\d{2}), timing (\d\.\d{2}), whiff miss (\d\.\d{2})\.",
          lambda s: stuff_by_member(s["text"])),
    Claim("membership changes",
          r"adding called strike lowers the composite's year-over-year correlation by (\d\.\d{3}) \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\] and alignment by "
          r"(\d\.\d{3}) \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\], while dropping whiff miss costs (\d\.\d{3}) \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\] "
          r"and dropping whiff (\d\.\d{3}) \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\]",
          lambda s: membership(s["membership"])),
    Claim("pitch model AUC by pitch type",
          r"AUC by pitch type: whiff (\d\.\d{2}) to (\d\.\d{2}), chase (\d\.\d{2}) to (\d\.\d{2}), weak contact (\d\.\d{2}) to (\d\.\d{2}), "
          r"ground ball (\d\.\d{2}) to (\d\.\d{2}), called strike (\d\.\d{2}) to (\d\.\d{2})",
          lambda s: auc_ranges(s["validation"])),
    Claim("pitch model log loss",
          r"whiff (\d\.\d{3}) against (\d\.\d{3}), chase (\d\.\d{3}) against (\d\.\d{3}), weak contact (\d\.\d{3}) against (\d\.\d{3}), "
          r"ground ball (\d\.\d{3}) against (\d\.\d{3})",
          lambda s: log_losses(s["validation"])),
    Claim("pitch model R-squared by pitch type",
          r"R-squared by pitch type: timing (\d\.\d{2}) to (\d\.\d{2}), whiff miss distance (\d\.\d{2}) to (\d\.\d{2}), horizontal alignment (\d\.\d{2}) to (\d\.\d{2})",
          lambda s: r2_ranges(s["validation"])),
    Claim("on-field gains",
          r"for strikeout rate \(cross-validated R² (\d\.\d{3}) to (\d\.\d{3}), gain ([+-]\d\.\d{3}) \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\]\) "
          r"and xwOBA allowed \((\d\.\d{3}) to (\d\.\d{3}), ([+-]\d\.\d{3}) \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\]\)",
          lambda s: on_field(s["outcome"])),
    Claim("on-field stability",
          r"year over year \((\d\.\d{2}) among (\d+) qualified pairs\) than the raw whiff rate it is built from \((\d\.\d{2})\) and Stuff\+ \((\d\.\d{2})\)",
          lambda s: stability(s["outcome"])),
    Claim("projection coefficients",
          r"with one season next = ([\d.]+) \+ ([\d.]+) x this, with two next = ([\d.]+) \+ ([\d.]+) x this \+ ([\d.]+) x previous",
          lambda s: projection_coefficients(s["projection"])),
    Claim("projection errors",
          r"the typical miss \(RMSE\) is ([\d.]+) points, against ([\d.]+) for using the season's score as the guess and ([\d.]+) for guessing 100",
          lambda s: projection_errors(s["projection"])),
    Claim("projection coverage", r"covers (\d+)% of outcomes", lambda s: [s["text"]["PROJ_COVER"].rstrip("%")]),
    Claim("driver survivors",
          r"\((\d) of (\d+) for whiff, (\d) for chase, (\d) for ground ball, (\d) for weak contact, (\d) for timing, (\d) for whiff miss\)",
          lambda s: survivors(s["drivers"])),
]


def check(readme: str, outputs: dict, claims: list[Claim] = CLAIMS) -> list[str]:
    """Every claim that is missing from the README, appears more than once, or disagrees with the outputs."""
    problems = []
    for claim in claims:
        try:
            expected = claim.expected(outputs)
        except KeyError as missing:
            problems.append(f"{claim.name}: output {missing} is not available, so it was not checked")
            continue
        matches = re.findall(claim.pattern, readme)
        if len(matches) != 1:
            problems.append(f"{claim.name}: found {len(matches)} matches for its pattern (expected 1); reword the README or update the pattern")
            continue
        found = list(matches[0]) if isinstance(matches[0], tuple) else [matches[0]]
        if found != expected:
            problems.append(f"{claim.name}: README says {found}, outputs say {expected}")
    return problems


def fix(readme: str, outputs: dict, claims: list[Claim] = CLAIMS) -> tuple[str, list[str]]:
    """The README with the figures of every claim that matches its pattern exactly once and disagrees with the outputs
    replaced by the outputs' figures (the sentence's words are never touched), and the names of the claims changed."""
    changed = []
    for claim in claims:
        try:
            expected = claim.expected(outputs)
        except KeyError:
            continue
        matches = list(re.finditer(claim.pattern, readme))
        if len(matches) != 1 or matches[0].re.groups != len(expected):
            continue
        match = matches[0]
        found = [match.group(i + 1) for i in range(match.re.groups)]
        if found == expected:
            continue
        for i in reversed(range(match.re.groups)):
            start, end = match.span(i + 1)
            readme = readme[:start] + expected[i] + readme[end:]
        changed.append(claim.name)
    return readme, changed


def main() -> int:
    if "--fix" in sys.argv:
        outputs = load_outputs()
        text, changed = fix(Path(README).read_text(encoding="utf-8"), outputs)
        Path(README).write_text(text, encoding="utf-8")
        print(f"updated the figures of {len(changed)} README claims: {changed}")
    outputs = load_outputs()
    if not outputs:
        print("No output files found; run the pipeline first.")
        return 1
    problems = check(Path(README).read_text(encoding="utf-8"), outputs, CLAIMS)
    for problem in problems:
        print("MISMATCH", problem)
    summary = f"{len(CLAIMS) - len(problems)} of {len(CLAIMS)} README claims match the outputs."
    print(summary)
    # the pipeline step reads this file to see that the check ran this time
    Path(REPORT).parent.mkdir(exist_ok=True)
    lines = ["MISMATCH " + p for p in problems] + [summary]
    Path(REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
