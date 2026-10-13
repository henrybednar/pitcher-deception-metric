"""
Build the Deception+ dashboard page from templates/dashboard.html.

  python build_artifact.py             standalone file with relative links (and the index.html copy)
  python build_artifact.py --artifact  fragment for the Claude artifact host

Reads artifact_data.json, outcome_validation.json and site_stats.json. The pitch-model validation, the Stuff+ detail,
the full component and on-field tables and the method live on the methodology page (build_methodology.py).
"""

import argparse
import json

from site_common import (
    DASHBOARD_FILE,
    INDEX_FILE,
    bin_table_rows,
    binned_means,
    json_for_script,
    load_stats,
    outcome_forest_rows,
    page_links,
    piece_dots_svg,
    render_page,
    reputation_rows,
    scatter_points,
    shared_head,
)

TITLE = "Deception+"
DESCRIPTION = (
    "Deception+ scores how far a pitcher's whiff, chase, weak contact, swing timing, and whiff miss "
    "distance results land above what the pitch's measured traits predict. Statcast {span}."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="store_true", help="emit an artifact fragment linking to published URLs")
    parser.add_argument("--out", default=None, help="default: the published page, or deception_dashboard_artifact.html with --artifact")
    args = parser.parse_args()

    with open("output/artifact_data.json", encoding="utf-8") as f:
        artifact_data = json.load(f)
    with open("output/outcome_validation.json", encoding="utf-8") as f:
        outcome_validation = json.load(f)
    stats = load_stats()
    links = page_links(args.artifact)
    points = scatter_points(artifact_data["points"])
    bins = binned_means([(p[2], p[3]) for p in points])

    core = round(100 * sum(1 for p in points if 90 <= p[3] < 110) / len(points))     # the script redraws this when it runs; this is the no-JavaScript text

    page = render_page(
        "templates/dashboard.html",
        title=TITLE,
        description=DESCRIPTION.format(span=stats["text"]["SEASON_SPAN"].replace(" to ", "-")),
        artifact_mode=args.artifact,
        text=stats["text"],
        page_file=DASHBOARD_FILE,
        replacements={
            "__SHARED_HEAD__": shared_head(),
            "__POINTS_JSON__": json_for_script(points),
            "__LORE_JSON__": json_for_script(artifact_data["lore"]),
            "__BINS_JSON__": json_for_script(bins),
            "__RULER_CORE_PCT__": f"{core}%",
            "__BIN_ROWS__": bin_table_rows(bins),
            "__PIECE_DOTS__": piece_dots_svg(stats["components"]),
            "__OUTCOME_FOREST_ROWS__": outcome_forest_rows(outcome_validation),
            "__REPUTATION_ROWS__": reputation_rows(artifact_data["lore"], links["leaderboard"]),
            "__LEADERBOARD_URL__": links["leaderboard"],
            "__METHODOLOGY_URL__": links["methodology"],
        },
    )
    out_path = args.out or ("deception_dashboard_artifact.html" if args.artifact else DASHBOARD_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out_path} ({'artifact' if args.artifact else 'standalone'}), {len(page):,} chars")
    if not args.artifact and not args.out:
        # the bare Pages address has no page of its own, so it serves a copy of the dashboard (its canonical link names the real one)
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"wrote {INDEX_FILE} (copy of the dashboard)")


if __name__ == "__main__":
    main()
