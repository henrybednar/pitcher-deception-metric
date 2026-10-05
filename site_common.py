"""
Shared helpers for build_artifact.py and build_leaderboard.py.

Templates hold the page markup. Two kinds of tokens are filled at build time:
  {{KEY}}        a text figure from site_stats.json (never typed into the HTML by hand)
  __TOKEN__      a JSON payload or structural block supplied by the build script

Two output modes:
  standalone  a complete HTML document with relative links between the pages
  artifact    a fragment for the Claude artifact host, linking to the published URLs
"""

import html
import json
import re
from pathlib import Path

DASHBOARD_FILE = "deception_dashboard.html"
LEADERBOARD_FILE = "deception_leaderboard.html"
ARTIFACT_URLS = {
    "dashboard": "https://claude.ai/artifact/8Tv4rsUay43bAfsr8A6aSJ",
    "leaderboard": "https://claude.ai/artifact/X1byK2MquLxXN44J9DrYwa",
}
FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%23141b22'/%3E"
    "%3Ctext x='16' y='23' font-family='Arial,sans-serif' font-size='20' font-weight='700' "
    "text-anchor='middle' fill='%23e8a33d'%3ED%2B%3C/text%3E%3C/svg%3E"
)
STANDALONE_BODY_STYLE = (
    "<style>html,body{margin:0;background:#f2f5f8}"
    "@media (prefers-color-scheme:dark){html,body{background:#10151c}}</style>"
)
TOKEN_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
PAYLOAD_PATTERN = re.compile(r"__[A-Z0-9_]+__")


def json_for_script(obj) -> str:
    """JSON that is safe inside an inline <script>: every `<` becomes a backslash-u-003c escape, so neither
    `</script>` nor `<!--` can appear in the page whatever the data holds, and the parser reads the same JSON."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")


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
        return {"dashboard": ARTIFACT_URLS["dashboard"], "leaderboard": ARTIFACT_URLS["leaderboard"]}
    return {"dashboard": DASHBOARD_FILE, "leaderboard": LEADERBOARD_FILE}


def head_block(title: str, description: str, artifact_mode: bool = False) -> str:
    if artifact_mode:
        # the artifact host supplies charset, viewport and the document skeleton
        return f"<title>{html.escape(title)}</title>\n"
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
    # one pass, so a payload that happens to contain another token's text is never expanded
    body = PAYLOAD_PATTERN.sub(lambda m: replacements.get(m.group(0), m.group(0)), body)
    head = head_block(title, description, artifact_mode)
    if artifact_mode:
        return head + body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n' + head + "</head>\n<body>\n"
        + STANDALONE_BODY_STYLE + "\n" + body + "\n</body>\n</html>\n"
    )


def component_table_rows(components: list) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for c in components:
        verdict = "Pass" if c["passes"] else "Fails"
        cls = "pass" if c["passes"] else "fail"
        total = "dx-row-total" if c["key"] == "composite" else ""
        in_score = "Yes" if c["in_score"] else "No"
        rows.append(
            f'<tr class="{total}"><th scope="row">{html.escape(c["label"])}</th>'
            f'<td>{esc(c["reliability"])}</td><td>{esc(c["yoy"])}</td><td>{esc(c["delta"])}</td><td>{esc(c["p"])}</td>'
            f'<td>{esc(c["n"])}</td><td class="dx-{cls}">{verdict}</td><td>{in_score}</td></tr>'
        )
    return "\n".join(rows)
