"""
Write the scores as a CSV, one row per scored pitcher-season, for anyone who wants to work with them.

  python build_scores_csv.py

Reads output/leaderboard_data.json and writes deception_scores.csv (UTF-8). It holds the Deception+ score, the five components
and ground ball with their samples and 95% intervals, the next-season projection, and the qualified flag. It leaves out Stuff+
(FanGraphs figures are not republished here) and the per-pitch-type detail.
"""

import csv
import io
import json

from site_common import SCORES_CSV_FILE

AREAS = ("whiff", "chase", "weak", "timing", "whiffmiss", "gb")
COLUMNS = (["player", "season", "role", "qualified", "deception_plus", "deception_plus_ci_lo", "deception_plus_ci_hi",
            "next_season_projection", "next_season_lo", "next_season_hi", "projection_seasons_used"]
           + [f"{area}_{field}" for area in AREAS for field in ("index", "n", "ci_lo", "ci_hi")])


def row_values(row: dict) -> list:
    """One CSV row from a leaderboard row: blank for a missing value, and the projection columns renamed."""
    renamed = {"player": row["name"], "next_season_projection": row.get("proj"), "next_season_lo": row.get("proj_lo"),
               "next_season_hi": row.get("proj_hi"), "projection_seasons_used": row.get("proj_basis")}
    values = [renamed[c] if c in renamed else row.get(c) for c in COLUMNS]
    counts = {c for c in COLUMNS if c.endswith("_n") or c == "projection_seasons_used"}
    return ["" if v is None else (int(v) if c in counts and float(v).is_integer() else v) for c, v in zip(COLUMNS, values)]


def rows_to_csv(rows: list) -> str:
    """The whole file as text: highest Deception+ first, pitcher-seasons without a score last."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in sorted(rows, key=lambda r: (r.get("deception_plus") is None, -(r.get("deception_plus") or 0))):
        writer.writerow(row_values(row))
    return out.getvalue()


def main() -> None:
    with open("output/leaderboard_data.json", encoding="utf-8") as f:
        rows = json.load(f)
    text = rows_to_csv(rows)
    with open(SCORES_CSV_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"wrote {SCORES_CSV_FILE}: {len(rows):,} pitcher-seasons, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
