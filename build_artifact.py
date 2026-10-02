"""
Build the Deception+ dashboard page from templates/dashboard.html.

  python build_artifact.py             standalone file with relative links
  python build_artifact.py --artifact  fragment for the Claude artifact host

Reads artifact_data.json, driver_analysis.json, stuffplus_relationship.json and site_stats.json.
"""

import argparse
import json

from site_common import (
    DASHBOARD_FILE,
    component_table_rows,
    load_stats,
    page_links,
    render_page,
)

TITLE = "Deception+"
DESCRIPTION = (
    "Deception+ scores how far a pitcher's whiff, chase, weak contact, swing timing, and whiff miss "
    "distance results land above what the pitch's measured traits predict. Statcast 2025-26."
)


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="store_true", help="emit an artifact fragment linking to published URLs")
    parser.add_argument("--out", default=DASHBOARD_FILE)
    args = parser.parse_args()

    with open("output/artifact_data.json", encoding="utf-8") as f:
        artifact_data = json.load(f)
    with open("output/driver_analysis.json", encoding="utf-8") as f:
        driver_data = json.load(f)
    with open("output/stuffplus_relationship.json", encoding="utf-8") as f:
        stuff_data = json.load(f)
    stats = load_stats()

    page = render_page(
        "templates/dashboard.html",
        title=TITLE,
        description=DESCRIPTION,
        artifact_mode=args.artifact,
        text=stats["text"],
        replacements={
            "__POINTS_JSON__": compact_json(artifact_data["points"]),
            "__LORE_JSON__": compact_json(artifact_data["lore"]),
            "__DRIVERS_JSON__": compact_json(driver_data),
            "__STUFFPLUS_JSON__": compact_json(stuff_data["outcomes"]),
            "__COMPONENT_ROWS__": component_table_rows(stats["components"]),
            "__LEADERBOARD_URL__": page_links(args.artifact)["leaderboard"],
        },
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out} ({'artifact' if args.artifact else 'standalone'}), {len(page):,} chars")


if __name__ == "__main__":
    main()
