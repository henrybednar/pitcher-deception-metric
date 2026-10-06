"""
Build the Deception+ methodology page from templates/methodology.html.

  python build_methodology.py             standalone file with relative links
  python build_methodology.py --artifact  fragment for the Claude artifact host

Reads driver_analysis.json, model_validation.json, pitch_type_scores.json and site_stats.json. It holds the
pitch-model validation, how a score is built, the membership and interval checks, the driver chart, the known
limits and the audit log; the dashboard keeps the headline checks.
"""

import argparse
import json

from site_common import METHODOLOGY_FILE, json_for_script, load_stats, page_links, pitch_type_rows, render_page, shared_head

TITLE = "Deception+ methodology"
DESCRIPTION = (
    "How Deception+ is built and checked: out-of-fold validation of the pitch models, the score's shrinkage and "
    "intervals, which components are in it and why, the known limits, and an audit log. Statcast 2024-26."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="store_true", help="emit an artifact fragment linking to published URLs")
    parser.add_argument("--out", default=None, help="default: the published page, or deception_methodology_artifact.html with --artifact")
    args = parser.parse_args()

    with open("output/driver_analysis.json", encoding="utf-8") as f:
        driver_data = json.load(f)
    with open("output/model_validation.json", encoding="utf-8") as f:
        validation_data = json.load(f)
    with open("output/pitch_type_scores.json", encoding="utf-8") as f:
        pitch_type_summary = json.load(f)
    stats = load_stats()
    links = page_links(args.artifact)

    page = render_page(
        "templates/methodology.html",
        title=TITLE,
        description=DESCRIPTION,
        artifact_mode=args.artifact,
        text=stats["text"],
        replacements={
            "__SHARED_HEAD__": shared_head(),
            "__DRIVERS_JSON__": json_for_script(driver_data),
            "__VALIDATION_JSON__": json_for_script(validation_data["outcomes"]),
            "__PITCHTYPE_ROWS__": pitch_type_rows(pitch_type_summary),
            "__DASHBOARD_URL__": links["dashboard"],
            "__LEADERBOARD_URL__": links["leaderboard"],
        },
    )
    out_path = args.out or ("deception_methodology_artifact.html" if args.artifact else METHODOLOGY_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out_path} ({'artifact' if args.artifact else 'standalone'}), {len(page):,} chars")


if __name__ == "__main__":
    main()
