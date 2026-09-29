"""
Build the Deception+ leaderboard page from templates/leaderboard.html.

  python build_leaderboard.py             standalone file with relative links
  python build_leaderboard.py --artifact  fragment for the Claude artifact host

Reads leaderboard_data.json and site_stats.json.
"""

import argparse
import json

from site_common import LEADERBOARD_FILE, load_stats, page_links, render_page

TITLE = "Deception+ Leaderboard"
DESCRIPTION = (
    "Sortable Deception+ scores and six component indexes for every scored pitcher-season, "
    "with 95% intervals and a starter or reliever filter. Statcast 2025-26."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="store_true", help="emit an artifact fragment linking to published URLs")
    parser.add_argument("--out", default=LEADERBOARD_FILE)
    args = parser.parse_args()

    with open("leaderboard_data.json", encoding="utf-8") as f:
        rows = json.load(f)
    stats = load_stats()

    page = render_page(
        "templates/leaderboard.html",
        title=TITLE,
        description=DESCRIPTION,
        artifact_mode=args.artifact,
        text=stats["text"],
        replacements={
            "__ROWS_JSON__": json.dumps(rows, separators=(",", ":"), ensure_ascii=False).replace("</", "<\/"),
            "__DASHBOARD_URL__": page_links(args.artifact)["dashboard"],
        },
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out} ({'artifact' if args.artifact else 'standalone'}), {len(page):,} chars")


if __name__ == "__main__":
    main()
