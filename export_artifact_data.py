"""
Pitcher Deception Project: export artifact_data.json from the canonical pitcher_season.csv
==========================================================================================
build_artifact.py reads this JSON. It holds the dashboard's scatter points
(one per pitcher-season with at least 100 swings) and eight hand-picked
pitchers for the reputation check.
"""

import json

import pandas as pd

HIGHLIGHT_NAMES = {
    "Sale, Chris", "Chapman, Aroldis", "Hader, Josh", "Rogers, Tyler",
    "Hill, Tim", "Ohtani, Shohei", "Duran, Jhoan", "Skubal, Tarik",
}
LORE_PICKS = [
    ("Skubal, Tarik", 2025), ("Ohtani, Shohei", 2026), ("Chapman, Aroldis", 2025),
    ("Hader, Josh", 2025), ("Duran, Jhoan", 2025), ("Rogers, Tyler", 2025),
    ("Hill, Tim", 2025), ("Sale, Chris", 2026),
]
MIN_SWINGS = 100

ps = pd.read_csv("output/pitcher_season.csv")
pop = ps[(ps["whiff_n"] >= MIN_SWINGS) & ps["deception_plus"].notna()].copy()

def pctile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


pop["whiff_pctile"] = pctile(pop["whiff_index"])
pop["gb_pctile"] = pctile(pop["gb_index"])
pop["dp_pctile"] = pctile(pop["deception_plus"])


def num(v):
    return None if pd.isna(v) else round(float(v), 1)


points = []
for _, r in pop.iterrows():
    points.append([
        r["player_name"], int(r["season"]), num(r["stuff_plus"]),
        num(r["whiff_index"]), num(r["chase_index"]), num(r["gb_index"]),
        num(r["weak_index"]), num(r["timing_index"]),
        num(r["deception_plus"]), int(r["whiff_n"]),
        r["player_name"] in HIGHLIGHT_NAMES, bool(r["qualified"]),
        num(r["whiffmiss_index"]),
    ])

lore = []
for name, season in LORE_PICKS:
    row = pop[(pop["player_name"] == name) & (pop["season"] == season)]
    if row.empty:
        print(f"WARNING: lore pick {name} {season} not found in population, skipping")
        continue
    r = row.iloc[0]
    lore.append({
        "name": name, "season": season,
        "whiff_index": num(r["whiff_index"]), "pctile": round(r["whiff_pctile"]),
        "gb_index": num(r["gb_index"]), "gb_pctile": round(r["gb_pctile"]),
        "deception_plus": num(r["deception_plus"]), "dp_pctile": round(r["dp_pctile"]),
        "n": int(r["whiff_n"]), "qualified": bool(r["qualified"]),
    })

with open("output/artifact_data.json", "w", encoding="utf-8") as f:
    json.dump({"points": points, "lore": lore}, f, separators=(",", ":"))

print(f"Saved artifact_data.json: {len(points):,} points, {len(lore)} lore rows.")
