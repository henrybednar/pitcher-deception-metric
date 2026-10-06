"""
Build the Deception+ dashboard page from templates/dashboard.html.

  python build_artifact.py             standalone file with relative links
  python build_artifact.py --artifact  fragment for the Claude artifact host

Reads artifact_data.json, driver_analysis.json, stuffplus_relationship.json, model_validation.json,
outcome_validation.json, pitch_type_scores.json and site_stats.json.
"""

import argparse
import json

from site_common import (
    DASHBOARD_FILE,
    component_table_rows,
    json_for_script,
    load_stats,
    outcome_coefficient_rows,
    outcome_table_rows,
    page_links,
    pitch_type_rows,
    render_page,
)

TITLE = "Deception+"
DESCRIPTION = (
    "Deception+ scores how far a pitcher's whiff, chase, weak contact, swing timing, and whiff miss "
    "distance results land above what the pitch's measured traits predict. Statcast 2024-26."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="store_true", help="emit an artifact fragment linking to published URLs")
    parser.add_argument("--out", default=None, help="default: the published page, or deception_dashboard_artifact.html with --artifact")
    args = parser.parse_args()

    with open("output/artifact_data.json", encoding="utf-8") as f:
        artifact_data = json.load(f)
    with open("output/driver_analysis.json", encoding="utf-8") as f:
        driver_data = json.load(f)
    with open("output/stuffplus_relationship.json", encoding="utf-8") as f:
        stuff_data = json.load(f)
    with open("output/model_validation.json", encoding="utf-8") as f:
        validation_data = json.load(f)
    with open("output/outcome_validation.json", encoding="utf-8") as f:
        outcome_validation = json.load(f)
    with open("output/pitch_type_scores.json", encoding="utf-8") as f:
        pitch_type_summary = json.load(f)
    stats = load_stats()

    page = render_page(
        "templates/dashboard.html",
        title=TITLE,
        description=DESCRIPTION,
        artifact_mode=args.artifact,
        text=stats["text"],
        replacements={
            "__POINTS_JSON__": json_for_script(artifact_data["points"]),
            "__LORE_JSON__": json_for_script(artifact_data["lore"]),
            "__DRIVERS_JSON__": json_for_script(driver_data),
            "__STUFFPLUS_JSON__": json_for_script(stuff_data["outcomes"]),
            "__VALIDATION_JSON__": json_for_script(validation_data["outcomes"]),
            "__COMPONENT_ROWS__": component_table_rows(stats["components"]),
            "__OUTCOME_ROWS__": outcome_table_rows(outcome_validation),
            "__OUTCOME_COEF_ROWS__": outcome_coefficient_rows(outcome_validation),
            "__PITCHTYPE_ROWS__": pitch_type_rows(pitch_type_summary),
            "__LEADERBOARD_URL__": page_links(args.artifact)["leaderboard"],
        },
    )
    out_path = args.out or ("deception_dashboard_artifact.html" if args.artifact else DASHBOARD_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out_path} ({'artifact' if args.artifact else 'standalone'}), {len(page):,} chars")


if __name__ == "__main__":
    main()
