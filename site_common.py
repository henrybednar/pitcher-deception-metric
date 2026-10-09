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
import re
from pathlib import Path
from urllib.parse import quote

DASHBOARD_FILE = "deception_dashboard.html"
LEADERBOARD_FILE = "deception_leaderboard.html"
METHODOLOGY_FILE = "deception_methodology.html"
STYLE_FILE = "templates/style.css"
ARTIFACT_URLS = {
    "dashboard": "https://claude.ai/artifact/8Tv4rsUay43bAfsr8A6aSJ",
    "leaderboard": "https://claude.ai/artifact/X1byK2MquLxXN44J9DrYwa",
    # no artifact for this page: the fragment links to the GitHub Pages copy
    "methodology": "https://henrybednar.github.io/pitcher-deception-metric/" + METHODOLOGY_FILE,
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


def head_block(title: str, description: str, artifact_mode: bool = False) -> str:
    if artifact_mode:
        # the artifact host supplies charset, viewport and the document skeleton; the body ground is still ours to set
        return f"<title>{html.escape(title)}</title>\n{BODY_STYLE}\n"
    desc = html.escape(description, quote=True)
    return (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f'<meta name="description" content="{desc}">\n'
        f'<meta property="og:title" content="{html.escape(title, quote=True)}">\n'
        f'<meta property="og:description" content="{desc}">\n'
        '<meta property="og:type" content="website">\n'
        f'<link rel="icon" href="{FAVICON}">\n'
    )


def render_page(template_path: str, *, title: str, description: str, artifact_mode: bool,
                replacements: dict, text: dict) -> str:
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
    head = head_block(title, description, artifact_mode)
    if artifact_mode:
        return head + body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n' + head + "</head>\n<body>\n"
        + STANDALONE_BODY_STYLE + "\n" + body + "\n</body>\n</html>\n"
    )


def format_gain(g: dict) -> str:
    return f"{g['gain']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]"


def gain_cell(g: dict) -> str:
    """A table cell for a gain: the gain on one line and its 95% interval under it, shaded when the interval is above zero."""
    shade = "dx-pass" if g["lo"] > 0 else ""
    return (f'<td class="{shade}"><span class="dx-gain">{g["gain"]:+.3f}</span>'
            f'<span class="dx-ci">[{g["lo"]:+.3f}, {g["hi"]:+.3f}]</span></td>')


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
    """Rows of the on-field validation table: for each outcome, cross-validated R2 of Stuff+ and Location+ and with Deception+
    added, then the same with last season's own result in the model, each gain with its 95% interval (shaded when the
    interval is above zero)."""
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for o in validation["outcomes"].values():
        r2 = o["r2"]
        g1, g2 = o["deception_over_stuff_location"], o["deception_over_own_stuff_location"]
        cells = [f'<th scope="row">{esc(o["label"])}</th>', f'<td>{r2["stuff_location"]:.3f}</td>', f'<td>{r2["stuff_location_deception"]:.3f}</td>',
                 gain_cell(g1),
                 f'<td>{r2["own_stuff_location"]:.3f}</td>', f'<td>{r2["own_stuff_location_deception"]:.3f}</td>',
                 gain_cell(g2)]
        rows.append("<tr>" + "".join(cells) + "</tr>")
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
    xwOBA) per standard deviation of each score, with the others held fixed, and the p-value."""
    esc = lambda value: html.escape(str(value), quote=True)
    k, x = validation["coefficients"]["k_pct"], validation["coefficients"]["xwoba"]
    rows = []
    for key, label in COEFFICIENT_ROWS:
        cells = [f'<th scope="row">{esc(label)}</th>',
                 f'<td class="{"dx-pass" if k[key]["p"] < 0.05 else ""}">{k[key]["coef"] * 100:+.2f} (p {esc(p_text(k[key]["p"]))})</td>',
                 f'<td class="{"dx-pass" if x[key]["p"] < 0.05 else ""}">{x[key]["coef"] * 1000:+.1f} (p {esc(p_text(x[key]["p"]))})</td>']
        rows.append("<tr>" + "".join(cells) + "</tr>")
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
            f'<td>{esc(c["reliability"])}</td><td>{esc(c["yoy"])}</td><td>{esc(c["delta"])}</td><td>{esc(c["p"])}</td>'
            f'<td>{esc(c["n"])}</td>'
            f'<td class="dx-badge-cell"><span class="dx-badge dx-badge-{cls}">{verdict}</span></td>'
            f'<td class="dx-badge-cell"><span class="dx-badge dx-badge-{in_cls}">{in_score}</span></td></tr>'
        )
    return "\n".join(rows)
