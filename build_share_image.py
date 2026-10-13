"""
Build the 1200 x 630 preview image that a shared link shows (LinkedIn, Slack, GitHub): the distribution of qualified
Deception+ scores with five pitchers marked, next to the name and the one-line definition.

  python build_share_image.py

Reads output/artifact_data.json (the dashboard's score ruler uses the same points) and writes share_image.png.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from site_common import SHARE_IMAGE_FILE  # noqa: E402

WIDTH, HEIGHT, DPI = 1200, 630, 100
BG, TEXT, TEXT_2, MUTED, RULE, ACCENT = "#f4f6f8", "#141b22", "#46535f", "#62707c", "#b4c0ca", "#a05b0c"
BAR, BAR_CORE = "#c4ccd3", "#9a9993"
LOW, HIGH, BIN = 60, 150, 5
CORE = (90, 110)                       # within one step (10 points) of average
PICKS = ["Hill, Tim", "Rogers, Tyler", "Duran, Jhoan", "Ohtani, Shohei", "Skubal, Tarik"]
URL = "henrybednar.github.io/pitcher-deception-metric"


def first_available(*names: str) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in available), "DejaVu Sans")


def draw(data: dict, path: str) -> tuple[int, int]:
    """Draw the share image from the dashboard's data ({"points": [...], "lore": [...]}) to `path`; returns the scores drawn and pitchers marked."""
    scores = np.array([p[8] for p in data["points"] if p[11] and p[8] is not None], dtype=float)
    lore = {p["name"]: p for p in data["lore"]}
    picks = sorted((lore[n] for n in PICKS if n in lore and lore[n]["deception_plus"] is not None), key=lambda p: p["deception_plus"])

    display = first_available("Bahnschrift", "Oswald", "Arial Narrow", "DejaVu Sans")
    body = first_available("Segoe UI", "Source Sans 3", "Arial", "DejaVu Sans")

    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)

    title = fig.text(0.06, 0.74, "Deception", fontsize=70, fontweight="bold", color=TEXT, family=display, va="bottom")
    fig.canvas.draw()
    plus_x = title.get_window_extent().x1 / WIDTH
    fig.text(plus_x, 0.74, "+", fontsize=70, fontweight="bold", color=ACCENT, family=display, va="bottom")
    fig.text(0.06, 0.70, "Pitcher results beyond what the\npitch's measured traits predict", fontsize=25, color=TEXT_2, family=body, va="top",
             linespacing=1.3)
    fig.text(0.06, 0.30, "Statcast pitch-level model, 2024 to 2026", fontsize=17, color=MUTED, family=body)
    fig.text(0.06, 0.245, f"{len(scores):,} qualified pitcher-seasons", fontsize=17, color=MUTED, family=body)
    fig.text(0.06, 0.09, URL, fontsize=16, color=ACCENT, family=body, fontweight="bold")

    ax = fig.add_axes([0.52, 0.2, 0.44, 0.5], facecolor=BG)
    edges = np.arange(LOW, HIGH + BIN, BIN)
    counts, _ = np.histogram(np.clip(scores, LOW, HIGH - 0.01), bins=edges)
    centers = edges[:-1] + BIN / 2
    colors = [BAR_CORE if CORE[0] <= e and e + BIN <= CORE[1] else BAR for e in edges[:-1]]
    ax.bar(centers, counts, width=BIN - 0.8, color=colors, zorder=2)
    top = counts.max()
    ax.set_xlim(LOW - 2, HIGH + 2)
    ax.set_ylim(0, top * 1.9)
    ax.axvline(100, ymin=0, ymax=0.62, color=MUTED, linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks([])
    ax.set_xticks(range(LOW, HIGH + 1, 10))
    ax.tick_params(axis="x", colors=MUTED, labelsize=13, length=4, color=RULE)
    for label in ax.get_xticklabels():
        label.set_family(body)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.set_xlabel("Deception+ (100 is average, 10 points is one step)", fontsize=14, color=TEXT_2, family=body, labelpad=10)

    for rank, p in enumerate(picks):
        x = p["deception_plus"]
        level = top * (1.18 + 0.23 * (rank % 3))
        ax.plot([x, x], [0, level], color=ACCENT, linewidth=2, zorder=3, solid_capstyle="round")
        ax.plot([x], [0], marker="o", markersize=9, color=ACCENT, markeredgecolor=BG, markeredgewidth=2, zorder=4, clip_on=False)
        text = ax.text(x, level + top * 0.05, f"{p['name'].split(', ')[0]} {x:.0f}", ha="right" if x > HIGH - 14 else "center",
                       va="bottom", fontsize=16, fontweight="bold", color=TEXT, family=body, zorder=5)
        text.set_path_effects([effects.withStroke(linewidth=4, foreground=BG)])

    fig.text(0.52, 0.765, "Where qualified pitcher-seasons land", fontsize=16, color=TEXT_2, family=body)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return len(scores), len(picks)


def main() -> None:
    with open("output/artifact_data.json", encoding="utf-8") as f:
        data = json.load(f)
    n_scores, n_marked = draw(data, SHARE_IMAGE_FILE)
    print(f"wrote {SHARE_IMAGE_FILE} ({WIDTH} x {HEIGHT}), {n_scores:,} scores, {n_marked} pitchers marked")


if __name__ == "__main__":
    main()
