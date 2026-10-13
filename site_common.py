"""
Shared helpers for build_artifact.py (dashboard), build_methodology.py and build_leaderboard.py.

Templates hold the page markup. Two kinds of tokens are filled at build time:
  {{KEY}}        a text figure from site_stats.json (never typed into the HTML by hand)
  __TOKEN__      a JSON payload or structural block supplied by the build script
The dashboard and methodology pages share templates/style.css, supplied as __SHARED_HEAD__.

Two output modes:
  standalone  a complete HTML document with relative links between the pages
  artifact    a fragment for the Claude artifact host, linking to the published URLs
"""

import html
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

PAGES_BASE = "https://henrybednar.github.io/pitcher-deception-metric/"
DASHBOARD_FILE = "deception_dashboard.html"
LEADERBOARD_FILE = "deception_leaderboard.html"
METHODOLOGY_FILE = "deception_methodology.html"
INDEX_FILE = "index.html"          # a copy of the dashboard, so the bare Pages address opens it
SHARE_IMAGE_FILE = "share_image.png"
SHARE_IMAGE_ALT = "Histogram of qualified pitcher-season Deception+ scores centered on 100, with five pitchers marked"
STYLE_FILE = "templates/style.css"
ARTIFACT_URLS = {
    "dashboard": "https://claude.ai/artifact/8Tv4rsUay43bAfsr8A6aSJ",
    "leaderboard": "https://claude.ai/artifact/X1byK2MquLxXN44J9DrYwa",
    # no artifact for this page: the fragment links to the GitHub Pages copy
    "methodology": PAGES_BASE + METHODOLOGY_FILE,
}
FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600&family=Source+Sans+3:wght@400;500;600&'
    'family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">\n'
)
FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%23141b22'/%3E"
    "%3Ctext x='16' y='23' font-family='Arial,sans-serif' font-size='20' font-weight='700' "
    "text-anchor='middle' fill='%23e8a33d'%3ED%2B%3C/text%3E%3C/svg%3E"
)
# the page ground matches --bg; an explicit light or dark choice in the artifact host wins over the system setting
BODY_STYLE = (
    "<style>html,body{margin:0;background:#f4f6f8}"
    '@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) body{background:#10151c}}'
    ':root[data-theme="dark"] body{background:#10151c}</style>'
)
STANDALONE_BODY_STYLE = BODY_STYLE
TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
PAYLOAD_PATTERN = re.compile(r"__[A-Z0-9_]+__")


def json_for_script(obj) -> str:
    """JSON that is safe inside an inline <script>: every `<` becomes a backslash-u-003c escape, so neither
    `</script>` nor `<!--` can appear in the page whatever the data holds, and the parser reads the same JSON.
    NaN and infinity are refused (they are not JSON, and would reach a chart as a missing or broken value)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")


def load_stats(path: str = "output/site_stats.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fill_stats(template: str, text: dict) -> str:
    """Replace every {{KEY}}; fail loudly on a key that has no value."""
    missing = sorted({k for k in TOKEN_PATTERN.findall(template) if k not in text})
    if missing:
        raise KeyError(f"template uses keys missing from site_stats.json: {missing}")
    return TOKEN_PATTERN.sub(lambda m: html.escape(str(text[m.group(1)]), quote=True), template)


def page_links(artifact_mode: bool) -> dict:
    if artifact_mode:
        return dict(ARTIFACT_URLS)
    return {"dashboard": DASHBOARD_FILE, "leaderboard": LEADERBOARD_FILE, "methodology": METHODOLOGY_FILE}


def shared_head() -> str:
    """Font links and the stylesheet the dashboard and methodology pages share (the __SHARED_HEAD__ payload)."""
    return FONT_LINKS + "<style>\n" + Path(STYLE_FILE).read_text(encoding="utf-8").rstrip() + "\n</style>"


def head_block(title: str, description: str, artifact_mode: bool = False, page_file: str | None = None) -> str:
    if artifact_mode:
        # the artifact host supplies charset, viewport and the document skeleton; the body ground is still ours to set
        return f"<title>{html.escape(title)}</title>\n{BODY_STYLE}\n"
    desc = html.escape(description, quote=True)
    name = html.escape(title, quote=True)
    image = PAGES_BASE + SHARE_IMAGE_FILE
    url = f'<meta property="og:url" content="{PAGES_BASE}{page_file}">\n<link rel="canonical" href="{PAGES_BASE}{page_file}">\n' if page_file else ""
    return (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f'<meta name="description" content="{desc}">\n'
        f'<meta property="og:title" content="{name}">\n'
        f'<meta property="og:description" content="{desc}">\n'
        '<meta property="og:type" content="website">\n'
        '<meta property="og:site_name" content="Deception+">\n'
        + url +
        f'<meta property="og:image" content="{image}">\n'
        '<meta property="og:image:width" content="1200">\n'
        '<meta property="og:image:height" content="630">\n'
        f'<meta property="og:image:alt" content="{html.escape(SHARE_IMAGE_ALT, quote=True)}">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:title" content="{name}">\n'
        f'<meta name="twitter:description" content="{desc}">\n'
        f'<meta name="twitter:image" content="{image}">\n'
        f'<link rel="icon" href="{FAVICON}">\n'
    )


def render_page(template_path: str, *, title: str, description: str, artifact_mode: bool,
                replacements: dict, text: dict, page_file: str | None = None) -> str:
    """Load a template, fill payload tokens and stats, and wrap for the chosen mode."""
    template = Path(template_path).read_text(encoding="utf-8")
    if "__HEAD__" not in template:
        raise ValueError(f"{template_path} has no __HEAD__ token")
    body = fill_stats(template.replace("__HEAD__\n", "", 1), text)
    for token in replacements:
        if token not in body:
            raise ValueError(f"{template_path} has no {token} token")
    unsupplied = sorted(set(PAYLOAD_PATTERN.findall(body)) - set(replacements))
    if unsupplied:
        raise ValueError(f"{template_path} has tokens with no replacement: {unsupplied}")
    # one pass, so a payload that happens to contain another token's text is never expanded
    body = PAYLOAD_PATTERN.sub(lambda m: replacements.get(m.group(0), m.group(0)), body)
    head = head_block(title, description, artifact_mode, page_file)
    if artifact_mode:
        return head + body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n' + head + "</head>\n<body>\n"
        + STANDALONE_BODY_STYLE + "\n" + body + "\n</body>\n</html>\n"
    )


FOREST_W, FOREST_H = 170, 22


def forest_svg(g: dict, lo_dom: float, hi_dom: float) -> str:
    """A dot (the gain) and a line (its 95% interval) on a scale shared by every row, with a dashed zero line.
    The mark turns green when the interval is above zero."""
    def x(v: float) -> float:
        return 4 + (v - lo_dom) / (hi_dom - lo_dom) * (FOREST_W - 8)

    mid = FOREST_H / 2
    label = f'gain {g["gain"]:+.3f}, 95% interval {g["lo"]:+.3f} to {g["hi"]:+.3f}'
    return (f'<svg class="dx-forest{" clear" if g["lo"] > 0 else ""}" viewBox="0 0 {FOREST_W} {FOREST_H}" role="img" aria-label="{label}">'
            f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="1" y2="{FOREST_H - 1}"/>'
            f'<line class="ci" x1="{x(g["lo"]):.1f}" x2="{x(g["hi"]):.1f}" y1="{mid}" y2="{mid}"/>'
            f'<circle class="pt" cx="{x(g["gain"]):.1f}" cy="{mid}" r="4.5"/></svg>')


def bar_cell(text: str, lo: float = 0.0, hi: float = 1.0) -> str:
    """A number with a thin bar under it showing where it sits between lo and hi (a correlation, so 0 to 1)."""
    esc_text = html.escape(str(text), quote=True)
    try:
        share = min(1.0, max(0.0, (float(text) - lo) / (hi - lo)))
    except ValueError:
        return f"<td>{esc_text}</td>"
    return f'<td><span class="dx-barcell">{esc_text}<span class="dx-minibar" aria-hidden="true"><i style="--v:{share:.3f}"></i></span></span></td>'


def pitch_type_rows(summary: dict) -> str:
    """Rows of the by-pitch-type table: for each pitch type, the pitcher-seasons scored, the median sample and the
    split-half reliability (Spearman-Brown corrected) of the whiff and chase scores."""
    from pitch_type_scores import PITCH_TYPE_NAMES   # imported here so the page builders do not load the scoring code unless they need a table

    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for pitch_type, per_label in summary["types"].items():
        cells = [f'<th scope="row">{esc(PITCH_TYPE_NAMES[pitch_type].capitalize())}</th>']
        for label in ("whiff", "chase"):
            r = per_label[label]
            reliability = "-" if r["reliability"] is None else f"{r['reliability']:.2f}"
            cells += [f'<td>{r["n_pitcher_seasons"]:,}</td>', f'<td>{r["median_n"]:.0f}</td>', f"<td>{reliability}</td>"]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def outcome_table_rows(validation: dict) -> str:
    """Rows of the on-field table: for each outcome, cross-validated R2 without and with Deception+ added and a forest plot of the gain
    (dot) and its 95% interval (line), first against Stuff+ and Location+ alone, then also with last season's own result."""
    esc = lambda value: html.escape(str(value), quote=True)
    outcomes = list(validation["outcomes"].values())
    gains = [g for o in outcomes for g in (o["deception_over_stuff_location"], o["deception_over_own_stuff_location"])]
    lo_dom, hi_dom = min(0.0, min(g["lo"] for g in gains)), max(g["hi"] for g in gains)
    pad = (hi_dom - lo_dom) * 0.06
    lo_dom, hi_dom = lo_dom - pad, hi_dom + pad

    def cell(before: float, after: float, g: dict) -> str:
        return (f'<td class="dx-forest-cell"><span class="dx-r2">R&sup2; {before:.3f} to {after:.3f}</span>'
                f'<span class="dx-forest-row">{forest_svg(g, lo_dom, hi_dom)}<span class="dx-gain">{g["gain"]:+.3f}</span></span></td>')

    rows = []
    for o in outcomes:
        r2 = o["r2"]
        rows.append(f'<tr><th scope="row">{esc(o["label"])}</th>'
                    + cell(r2["stuff_location"], r2["stuff_location_deception"], o["deception_over_stuff_location"])
                    + cell(r2["own_stuff_location"], r2["own_stuff_location_deception"], o["deception_over_own_stuff_location"]) + "</tr>")
    return "\n".join(rows)


BIN_GROUPS = 10          # equal-sized Stuff+ groups the scatter averages over


def scatter_points(points: list) -> list:
    """The qualified pitcher-seasons that have a score and a Stuff+ figure, slimmed to what the dashboard draws:
    [name, season, Stuff+, Deception+, swings, has a reputation for an unusual delivery]."""
    return [[p[0], p[1], p[2], p[8], p[9], p[10]] for p in points if p[11] and p[8] is not None and p[2] is not None]


def binned_means(rows: list, groups: int = BIN_GROUPS) -> list:
    """Mean of y, with its 95% interval, for each of `groups` equal-sized slices of the (x, y) pairs ordered by x. Slices under 20 are dropped."""
    ordered = sorted(rows)
    size = -(-len(ordered) // groups)
    bins = []
    for start in range(0, len(ordered), size):
        chunk = ordered[start:start + size]
        n = len(chunk)
        if n < 20:
            continue
        mx, my = sum(x for x, _ in chunk) / n, sum(y for _, y in chunk) / n
        half = 1.96 * math.sqrt(sum((y - my) ** 2 for _, y in chunk) / (n - 1) / n)
        bins.append({"x": round(mx, 2), "y": round(my, 2), "lo": round(my - half, 2), "hi": round(my + half, 2), "n": n})
    return bins


def bin_table_rows(bins: list) -> str:
    """The binned means as table rows, for readers without JavaScript."""
    rows = []
    for i, b in enumerate(bins, 1):
        name = f"{i} (lowest)" if i == 1 else (f"{i} (highest)" if i == len(bins) else str(i))
        rows.append(f'<tr><th scope="row">{name}</th><td>{b["x"]:.0f}</td><td>{b["y"]:.1f}</td><td>{b["lo"]:.1f} to {b["hi"]:.1f}</td><td>{b["n"]}</td></tr>')
    return "\n".join(rows)


DOTS_W = 352


def piece_dots_svg(components: list) -> str:
    """A dot plot of the pieces in the score, sorted by split-half reliability: a filled dot for the correlation within a season, an
    open dot for the correlation with the next season, the two values at the right, and the composite on its own row at the bottom."""
    esc = lambda value: html.escape(str(value), quote=True)
    pieces = sorted((c for c in components if c["in_score"] and c["key"] != "composite"), key=lambda c: -float(c["reliability"]))
    total = next((c for c in components if c["key"] == "composite"), None)
    rows = [(c["label"], float(c["reliability"]), float(c["yoy"]), False) for c in pieces]
    if total:
        rows.append(("Deception+", float(total["reliability"]), float(total["yoy"]), True))
    x0, x1, col_a, col_b, top, step = 128, 236, 288, 342, 34, 28
    height = top + step * len(rows) + 30

    def x(v: float) -> float:
        return x0 + max(0.0, min(1.0, v)) * (x1 - x0)

    aria = "Correlation within a season and with the next season, by piece: " + "; ".join(f"{n} {a:.3f} and {b:.3f}" for n, a, b, _ in rows)
    out = [f'<svg class="dx-dots" viewBox="0 0 {DOTS_W} {height}" role="img" aria-label="{esc(aria)}">',
           '<circle class="a" cx="6" cy="10" r="5"/><text class="hdr" x="16" y="14">within season</text>',
           '<circle class="b" cx="106" cy="10" r="5"/><text class="hdr" x="116" y="14">next season</text>',
           f'<text class="hdr" x="{col_a}" y="14" text-anchor="end">within</text><text class="hdr" x="{col_b}" y="14" text-anchor="end">next</text>']
    bottom = top + step * len(rows)
    for tick in (0.0, 0.5, 1.0):
        out.append(f'<line class="grid" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 2}" y2="{bottom - 2}"/>'
                   f'<text class="tick" x="{x(tick):.1f}" y="{bottom + 14}" text-anchor="middle">{tick:g}</text>')
    for i, (name, a, b, is_total) in enumerate(rows):
        y = top + i * step + 12
        if is_total:
            out.append(f'<line class="sep" x1="0" x2="{DOTS_W}" y1="{y - 14}" y2="{y - 14}"/>')
        cls = "lbl tot" if is_total else "lbl"
        out.append(f'<text class="{cls}" x="0" y="{y + 4}">{esc(name[:1].upper() + name[1:])}</text>'
                   f'<line class="gap" x1="{x(min(a, b)):.1f}" x2="{x(max(a, b)):.1f}" y1="{y}" y2="{y}"/>'
                   f'<circle class="a" cx="{x(a):.1f}" cy="{y}" r="5.5"/><circle class="b" cx="{x(b):.1f}" cy="{y}" r="5"/>'
                   f'<text class="val{" tot" if is_total else ""}" x="{col_a}" y="{y + 4}" text-anchor="end">{a:.3f}</text>'
                   f'<text class="val{" tot" if is_total else ""}" x="{col_b}" y="{y + 4}" text-anchor="end">{b:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def row_label(label: str) -> str:
    """A label starting a table row: capitalized, except a stat name that starts lower on purpose (xwOBA, wOBA)."""
    first = label.split(" ")[0]
    if first[:1].islower() and any(ch.isupper() for ch in first[1:]):
        return label
    return label[:1].upper() + label[1:]


def outcome_forest_rows(validation: dict) -> str:
    """Rows of the dashboard's on-field chart: one forest plot per test (beyond Stuff+ and Location+, then also beyond last season's
    own result), outcomes that clear zero first, outcomes that clear in neither test muted."""
    esc = lambda value: html.escape(str(value), quote=True)
    first, own = "deception_over_stuff_location", "deception_over_own_stuff_location"
    outcomes = list(validation["outcomes"].values())
    gains = [o[k] for o in outcomes for k in (first, own)]
    lo_dom, hi_dom = min(0.0, min(g["lo"] for g in gains)), max(g["hi"] for g in gains)
    pad = (hi_dom - lo_dom) * 0.06
    lo_dom, hi_dom = lo_dom - pad, hi_dom + pad

    def cell(g: dict) -> str:
        return (f'<td class="dx-forest-cell"><span class="dx-forest-row">{forest_svg(g, lo_dom, hi_dom)}'
                f'<span class="dx-gain-stack"><span class="dx-gain">{g["gain"]:+.3f}</span>'
                f'<span class="dx-ci">[{g["lo"]:+.3f}, {g["hi"]:+.3f}]</span></span></span></td>')

    rows = []
    for o in sorted(outcomes, key=lambda o: (o[first]["lo"] <= 0, -o[first]["gain"])):
        muted = o[first]["lo"] <= 0 and o[own]["lo"] <= 0
        label = row_label(o["label"])
        rows.append(f'<tr class="{"dx-muted" if muted else ""}"><th scope="row">{esc(label)}</th>{cell(o[first])}{cell(o[own])}</tr>')
    return "\n".join(rows)


def ordinal(n: int) -> str:
    n = int(n)
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def percentile_cell(value, pct) -> str:
    """A table cell with a score, its percentile and a thin bar showing where that percentile sits."""
    if value is None:
        return "<td>n/a</td>"
    return (f'<td><span class="dx-pcell">{value:.1f} ({ordinal(pct)})'
            f'<span class="dx-pbar" aria-hidden="true"><i style="--p:{int(pct)}"></i></span></span></td>')


def reputation_rows(lore: list, leaderboard_url: str) -> str:
    """Rows of the reputation table. Each name links to the leaderboard already searching for that pitcher, so the
    table needs no script and the reader can go from a name to every season of it."""
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for p in lore:
        last, _, first = p["name"].partition(", ")
        href = f"{leaderboard_url}#q={quote(p['name'])}"
        rows.append(
            f'<tr><th scope="row"><a class="dx-link" href="{esc(href)}">{esc(f"{first} {last}".strip())}</a></th>'
            f'<td>{esc(p["season"])}</td><td>{esc(p["n"])}</td>'
            + percentile_cell(p["deception_plus"], p["dp_pctile"]) + percentile_cell(p["whiff_index"], p["pctile"])
            + percentile_cell(p["gb_index"], p["gb_pctile"]) + "</tr>")
    return "\n".join(rows)


def p_text(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


COEFFICIENT_ROWS = [("stuff_plus", "Stuff+"), ("location_plus", "Location+"), ("whiff_index", "Whiff"), ("chase_index", "Chase"),
                    ("weak_index", "Weak contact"), ("timing_index", "Timing"), ("whiffmiss_index", "Whiff miss distance")]


def outcome_coefficient_rows(validation: dict) -> str:
    """Rows of the member table: the change in the next season's strikeout rate (percentage points) and xwOBA allowed (points of
    xwOBA) per standard deviation of each score, with the others held fixed, and the p-value under each, shaded when p < 0.05."""
    esc = lambda value: html.escape(str(value), quote=True)
    k, x = validation["coefficients"]["k_pct"], validation["coefficients"]["xwoba"]

    def cell(c: dict, scale: int, digits: int) -> str:
        return (f'<td class="{"dx-sig" if c["p"] < 0.05 else ""}"><span class="dx-gain">{c["coef"] * scale:+.{digits}f}</span>'
                f'<span class="dx-ci">p {esc(p_text(c["p"]))}</span></td>')

    rows = []
    for key, label in COEFFICIENT_ROWS:
        rows.append(f'<tr><th scope="row">{esc(label)}</th>' + cell(k[key], 100, 2) + cell(x[key], 1000, 1) + "</tr>")
    return "\n".join(rows)


def component_table_rows(components: list) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for c in components:
        verdict, cls = ("Pass", "pass") if c["passes"] else ("Fails", "fail")
        total = "dx-row-total" if c["key"] == "composite" else ""
        in_score, in_cls = ("In the score", "yes") if c["in_score"] else ("Left out", "no")
        rows.append(
            f'<tr class="{total}"><th scope="row">{html.escape(c["label"])}</th>'
            + bar_cell(c["reliability"]) + bar_cell(c["yoy"]) + f'<td>{esc(c["delta"])}</td>'
            f'<td class="dx-badge-cell"><span class="dx-badge dx-badge-{cls}">{verdict}</span>'
            f'<span class="dx-ci">p {esc(c["p"])} &middot; n {esc(c["n"])}</span></td>'
            f'<td class="dx-badge-cell"><span class="dx-badge dx-badge-{in_cls}">{in_score}</span></td></tr>'
        )
    return "\n".join(rows)
