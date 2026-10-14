import json

import pytest

import site_common as sc


def test_json_for_script_never_emits_a_less_than_sign_and_round_trips():
    data = {"name": "</script><script>alert(1)</script>", "note": "<!-- x -->", "n": [1, 2.5, None]}

    text = sc.json_for_script(data)

    assert "<" not in text
    assert json.loads(text) == data


def test_payload_tokens_are_replaced_in_one_pass_so_data_cannot_inject_another_token(tmp_path):
    template = tmp_path / "t.html"
    template.write_text("__HEAD__\n<p>{{A}}</p><script>var x = __ROWS__; var y = __OTHER__;</script>", encoding="utf-8")

    page = sc.render_page(str(template), title="t", description="d", artifact_mode=True, text={"A": "1"},
                          replacements={"__ROWS__": sc.json_for_script(["__OTHER__"]), "__OTHER__": "42"})

    assert page.count("__OTHER__") == 1          # the one inside the data survived as data
    assert "var y = 42" in page


def test_render_page_rejects_a_payload_token_the_template_does_not_have(tmp_path):
    template = tmp_path / "t.html"
    template.write_text("__HEAD__\n<p>hi</p>", encoding="utf-8")

    with pytest.raises(ValueError):
        sc.render_page(str(template), title="t", description="d", artifact_mode=True, text={}, replacements={"__ROWS__": "[]"})


def test_fill_stats_escapes_text_for_attributes_as_well_as_text_nodes_and_rejects_missing_keys():
    out = sc.fill_stats('<div aria-label="{{N}}">{{N}}</div>', {"N": 'x" onfocus="alert(1)'})

    assert 'onfocus="' not in out and "&quot;" in out
    with pytest.raises(KeyError):
        sc.fill_stats("{{MISSING}}", {})


def test_the_component_table_escapes_every_cell():
    row = {"key": "whiff", "label": "<b>Whiff</b>", "passes": True, "in_score": True, "reliability": "0.610", "yoy": "0.509",
           "delta": "+0.065", "p": "<0.0001", "n": 1093}

    html_rows = sc.component_table_rows([row])

    assert "<b>Whiff</b>" not in html_rows and "&lt;b&gt;" in html_rows
    assert "p <0.0001" not in html_rows and "p &lt;0.0001" in html_rows


def test_pitch_type_rows_list_each_type_with_the_sample_and_reliability_of_whiff_and_chase():
    from site_common import pitch_type_rows

    summary = {"types": {
        "SL": {"whiff": {"reliability": 0.641, "n_pitcher_seasons": 1084, "median_n": 98.0},
               "chase": {"reliability": None, "n_pitcher_seasons": 1156, "median_n": 104.0}},
    }}

    row = pitch_type_rows(summary)

    assert row.startswith('<tr><th scope="row">Sliders</th>')
    assert "<td>1,084</td><td>98</td><td>0.64</td>" in row
    assert "<td>1,156</td><td>104</td><td>-</td>" in row                      # a reliability that could not be estimated reads as a dash


def test_json_for_script_refuses_nan_and_infinity():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sc.json_for_script({"x": [1.0, bad]})


def test_render_page_rejects_a_template_token_nobody_supplied(tmp_path):
    template = tmp_path / "t.html"
    template.write_text("__HEAD__\n<script>var rows = __ROWS_JSON__; var more = __OTHER_JSON__;</script>", encoding="utf-8")

    with pytest.raises(ValueError, match="__OTHER_JSON__"):
        sc.render_page(str(template), title="t", description="d", artifact_mode=True, text={}, replacements={"__ROWS_JSON__": "[]"})


def test_the_artifact_fragment_does_not_default_to_the_published_pages_file(tmp_path, monkeypatch):
    import build_artifact
    import build_leaderboard

    for module, published, fragment in ((build_artifact, "DASHBOARD_FILE", "deception_dashboard_artifact.html"),
                                        (build_leaderboard, "LEADERBOARD_FILE", "deception_leaderboard_artifact.html")):
        source = open(module.__file__, encoding="utf-8").read()
        assert fragment in source and f'else {published})' in source


def test_the_artifact_fragment_of_the_methodology_page_does_not_default_to_the_published_file():
    import build_methodology

    source = open(build_methodology.__file__, encoding="utf-8").read()

    assert "deception_methodology_artifact.html" in source and "else METHODOLOGY_FILE)" in source


def test_page_links_name_all_three_pages_in_both_modes():
    standalone, artifact = sc.page_links(False), sc.page_links(True)

    assert set(standalone) == set(artifact) == {"dashboard", "leaderboard", "methodology", "csv"}
    assert standalone["csv"] == sc.SCORES_CSV_FILE and artifact["csv"] == sc.PAGES_BASE + sc.SCORES_CSV_FILE      # a download cannot start inside the artifact frame
    assert standalone["methodology"] == sc.METHODOLOGY_FILE
    assert artifact["methodology"].endswith(sc.METHODOLOGY_FILE) and artifact["methodology"].startswith("https://")


def test_shared_head_carries_the_fonts_and_the_stylesheet_both_pages_use():
    head = sc.shared_head()

    assert head.startswith("<link") and "fonts.googleapis.com" in head
    assert ".dxroot {" in head and head.rstrip().endswith("</style>")
    for template in ("templates/dashboard.html", "templates/methodology.html"):
        text = open(template, encoding="utf-8").read()
        assert "__SHARED_HEAD__" in text and "<style>" not in text      # the stylesheet lives in one file


def test_every_methodology_anchor_the_other_pages_link_to_exists():
    import re

    methodology = open("templates/methodology.html", encoding="utf-8").read()
    ids = set(re.findall(r'\bid="([^"]+)"', methodology))
    linked = set()
    for template in ("templates/dashboard.html", "templates/leaderboard.html", "templates/methodology.html"):
        text = open(template, encoding="utf-8").read()
        linked |= set(re.findall(r'__METHODOLOGY_URL__#([a-z-]+)', text))
        if template.endswith("methodology.html"):
            linked |= set(re.findall(r'href="#([a-z-]+)"', text))

    assert linked, "no anchors found, so the test checks nothing"
    assert linked <= ids, sorted(linked - ids)


def test_each_page_links_to_the_other_two():
    expected = {"templates/dashboard.html": ("__LEADERBOARD_URL__", "__METHODOLOGY_URL__"),
                "templates/methodology.html": ("__DASHBOARD_URL__", "__LEADERBOARD_URL__"),
                "templates/leaderboard.html": ("__DASHBOARD_URL__", "__METHODOLOGY_URL__")}
    for template, tokens in expected.items():
        text = open(template, encoding="utf-8").read()
        for token in tokens:
            assert token in text, (template, token)


def test_a_forest_mark_turns_green_only_when_its_interval_is_above_zero_and_sits_on_the_shared_scale():
    clear = sc.forest_svg({"gain": 0.071, "lo": 0.032, "hi": 0.110}, -0.02, 0.12)
    unclear = sc.forest_svg({"gain": -0.002, "lo": -0.010, "hi": 0.004}, -0.02, 0.12)

    assert 'class="dx-forest clear"' in clear and 'class="dx-forest"' in unclear
    assert "gain +0.071, 95% interval +0.032 to +0.110" in clear                 # the mark is described for a screen reader
    assert float(clear.split('class="zero" x1="')[1].split('"')[0]) == float(unclear.split('class="zero" x1="')[1].split('"')[0])   # zero sits in the same place on every row


def test_the_on_field_rows_hold_two_forest_plots_each_with_the_r2_before_and_after():
    gains = {"gain": 0.05, "lo": 0.01, "hi": 0.09}
    validation = {"outcomes": {"k_pct": {"label": "strikeout rate", "r2": {"stuff_location": 0.2, "stuff_location_deception": 0.25,
                                                                          "own_stuff_location": 0.4, "own_stuff_location_deception": 0.41},
                                         "deception_over_stuff_location": gains, "deception_over_own_stuff_location": {"gain": 0.01, "lo": -0.01, "hi": 0.03}}}}

    row = sc.outcome_table_rows(validation)

    assert row.count("<svg") == 2 and row.count("dx-forest clear") == 1
    assert "R&sup2; 0.200 to 0.250" in row and "R&sup2; 0.400 to 0.410" in row


def test_the_coefficient_table_shows_the_p_value_under_each_value_and_marks_only_the_significant_ones():
    cell = {"coef": 0.0123, "p": 0.0004}
    quiet = {"coef": 0.001, "p": 0.4}
    validation = {"coefficients": {"k_pct": {key: (cell if key == "whiff_index" else quiet) for key, _ in sc.COEFFICIENT_ROWS},
                                   "xwoba": {key: quiet for key, _ in sc.COEFFICIENT_ROWS}}}

    rows = sc.outcome_coefficient_rows(validation).splitlines()
    whiff = next(r for r in rows if ">Whiff<" in r)

    assert '<td class="dx-sig"><span class="dx-gain">+1.23</span><span class="dx-ci">p &lt;0.001</span>' in whiff
    assert whiff.count("dx-sig") == 1


def test_a_correlation_cell_draws_a_bar_scaled_from_zero_to_one_and_survives_text():
    assert "--v:0.614" in sc.bar_cell("0.614")
    assert "--v:1.000" in sc.bar_cell("1.7") and "--v:0.000" in sc.bar_cell("-0.2")      # clamped
    assert sc.bar_cell("n/a") == "<td>n/a</td>"


def test_the_component_table_marks_pass_and_status_with_labelled_badges_not_colour_alone():
    base = {"key": "gb", "label": "Ground ball", "reliability": "0.446", "yoy": "0.353", "delta": "+0.030", "p": "<0.0001", "n": 913}

    kept = sc.component_table_rows([{**base, "passes": True, "in_score": True}])
    left_out = sc.component_table_rows([{**base, "passes": False, "in_score": False}])

    assert "dx-badge-pass" in kept and ">Pass<" in kept and "dx-badge-yes" in kept and ">In the score<" in kept
    assert "dx-badge-fail" in left_out and ">Fails<" in left_out and "dx-badge-no" in left_out and ">Left out<" in left_out


def test_ordinals_handle_the_teens_and_the_100th():
    assert [sc.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 64, 100)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "64th", "100th"]


def test_reputation_rows_link_each_name_to_the_leaderboard_search_and_escape_it():
    lore = [{"name": 'Skubal, <Tarik>', "season": 2025, "n": 1552, "deception_plus": 145.3, "dp_pctile": 100,
             "whiff_index": 138.5, "pctile": 100, "gb_index": None, "gb_pctile": 3}]

    row = sc.reputation_rows(lore, "deception_leaderboard.html")

    assert 'href="deception_leaderboard.html#q=Skubal%2C%20%3CTarik%3E"' in row
    assert "&lt;Tarik&gt; Skubal" in row and "<Tarik>" not in row
    assert "145.3 (100th)" in row and "--p:100" in row and "<td>n/a</td>" in row      # a missing ground-ball index reads n/a, not nan


def test_the_leaderboard_opens_on_a_pitcher_when_linked_with_a_hash():
    text = open("templates/leaderboard.html", encoding="utf-8").read()

    assert "#q=(.+)$" in text and "qualifiedOnly').checked = false" in text


def test_each_page_starts_with_its_h1_and_has_main_and_footer_landmarks():
    import re

    for template in ("templates/dashboard.html", "templates/methodology.html", "templates/leaderboard.html"):
        text = open(template, encoding="utf-8").read()
        headings = re.findall(r"<(h[1-6])\b", text)

        assert headings[0] == "h1" and headings.count("h1") == 1, template
        assert "<main " in text and "</main>" in text and "<footer " in text and 'href="#main"' in text, template


def test_the_leaderboard_shares_the_stylesheet_and_its_filters_are_labelled_pill_groups():
    text = open("templates/leaderboard.html", encoding="utf-8").read()

    assert "__SHARED_HEAD__" in text and 'class="dxroot lb"' in text and "lbroot" not in text
    for group in ("seasonFilter", "roleFilter"):
        assert f'id="{group}"' in text and f'bindGroup(\'{group}\'' in text
    assert 'id="resetFilters"' in text and "aria-live" in text            # a reset control and an announced row count
    assert "season-tag" in text                                           # the season stays visible beside the name on a phone


def test_the_leaderboard_builder_supplies_the_shared_head():
    import build_leaderboard

    assert "shared_head()" in open(build_leaderboard.__file__, encoding="utf-8").read()


def test_the_artifact_head_names_the_page_first_and_sets_the_page_ground_for_both_themes():
    head = sc.head_block("Deception+", "d", artifact_mode=True)

    assert head.startswith("<title>Deception+</title>")                    # the host reads the title from the first 8KB
    assert "#f4f6f8" in head and '[data-theme="dark"] body' in head and "prefers-color-scheme:dark" in head


def test_the_dashboard_explains_each_statistic_it_shows_and_every_in_page_link_resolves():
    import re

    text = open("templates/dashboard.html", encoding="utf-8").read()
    key = text[text.index('id="stats-key"'):text.index("</details>", text.index('id="stats-key"'))]
    terms = re.findall(r"<dt>(.*?)</dt>", key)

    for term in ("r (correlation)", "Reliability", "R&sup2;", "Forecast gain", "95% interval", "Percentile", "AUC", "Calibration slope", "Shrinkage",
                 "Bootstrap", "Design effect", "Out of fold", "p-value", "Clustered on pitcher", "Qualified pitcher-season", "Standard deviation"):
        assert any(term in t for t in terms), term
    assert len(terms) == len(set(terms))                                      # no term is defined twice
    assert all(text.count(f'href="#{g}"') == 1 and f'id="{g}"' in key for g in re.findall(r'href="#(key-[\w-]+)"', key))     # each group has a jump link and a heading
    assert not re.search(r"[–—]", key)                              # public copy uses no dashes
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    assert {a for a in re.findall(r'href="#([\w-]+)"', text)} <= ids


def test_the_model_fit_ranges_are_also_published_as_percentages_for_the_plain_language_lines():
    from export_site_stats import validation_text

    def outcome(key, metric, values):
        return key, {"by_pitch_type": [{metric: v} for v in values], "calibration": {"slope": 1.0}}

    validation = {"outcomes": dict([
        outcome("gb", "auc", [0.62, 0.67]), outcome("calledstrike", "auc", [0.98, 0.99]),
        outcome("timing", "r2", [0.054, 0.206]), outcome("whiffmiss", "r2", [0.061, 0.394]),
        outcome("whiff", "auc", [0.757, 0.862]), outcome("chase", "auc", [0.81, 0.884]), outcome("weak", "auc", [0.68, 0.75])])}

    text = validation_text(validation)

    assert text["VAL_AUC_WHIFF_PCT"] == "76% to 86%" and text["VAL_AUC_CHASE_PCT"] == "81% to 88%"
    assert text["VAL_R2_TIMING_PCT"] == "5% to 21%" and text["VAL_R2_WHIFFMISS_PCT"] == "6% to 39%"


def test_the_on_field_example_reads_the_first_row_in_words():
    from export_site_stats import ov_example

    first = {"label": "Strikeout rate", "r2": {"stuff_location": 0.233, "stuff_location_deception": 0.304},
             "deception_over_stuff_location": {"gain": 0.071, "lo": 0.032, "hi": 0.110}}

    assert ov_example(first) == ("strikeout rate goes from 23% explained to 30% with Deception+ added, "
                                 "a gain of 7.1 points (95% interval 3.2 to 11.0)")


def test_a_standalone_page_head_carries_the_tags_a_shared_link_preview_needs():
    head = sc.head_block("Deception+", "A description", artifact_mode=False, page_file=sc.DASHBOARD_FILE)

    image = sc.PAGES_BASE + sc.SHARE_IMAGE_FILE
    assert f'<meta property="og:image" content="{image}">' in head                     # absolute, because the crawler fetches it from elsewhere
    assert '<meta property="og:image:width" content="1200">' in head and '<meta property="og:image:height" content="630">' in head
    assert '<meta name="twitter:card" content="summary_large_image">' in head
    assert f'<link rel="canonical" href="{sc.PAGES_BASE}{sc.DASHBOARD_FILE}">' in head
    assert f'<meta property="og:url" content="{sc.PAGES_BASE}{sc.DASHBOARD_FILE}">' in head


def test_the_head_skips_the_page_address_when_no_page_is_named_and_the_artifact_head_stays_minimal():
    plain = sc.head_block("Deception+", "d", artifact_mode=False)
    artifact = sc.head_block("Deception+", "d", artifact_mode=True)

    assert "og:url" not in plain and "canonical" not in plain and "og:image" in plain
    assert "og:image" not in artifact and artifact.startswith("<title>")


def test_the_dashboard_build_also_writes_the_index_copy_and_every_builder_names_its_page():
    import build_artifact
    import build_leaderboard
    import build_methodology

    source = open(build_artifact.__file__, encoding="utf-8").read()
    assert "INDEX_FILE" in source and "if not args.artifact and not args.out" in source
    for module, constant in ((build_artifact, "DASHBOARD_FILE"), (build_leaderboard, "LEADERBOARD_FILE"), (build_methodology, "METHODOLOGY_FILE")):
        assert f"page_file={constant}" in open(module.__file__, encoding="utf-8").read(), module.__name__


def test_the_share_image_is_1200_by_630_and_marks_only_the_pitchers_it_has(tmp_path):
    import numpy as np
    from PIL import Image

    import build_share_image

    rng = np.random.default_rng(0)
    points = [[f"P{i}", 2025, 100, 100, 100, 100, 100, 100, float(s), 500, False, True, 100] for i, s in enumerate(rng.normal(100, 10, 300))]
    lore = [{"name": "Skubal, Tarik", "deception_plus": 145.3}, {"name": "Hill, Tim", "deception_plus": 90.0}, {"name": "Nobody, Else", "deception_plus": 120.0}]
    out = tmp_path / "card.png"

    n_scores, n_marked = build_share_image.draw({"points": points, "lore": lore}, str(out))

    assert (n_scores, n_marked) == (300, 2)                                            # the unlisted name is not drawn
    assert Image.open(out).size == (1200, 630)


def test_the_pipeline_page_step_builds_the_index_copy_and_the_share_image_and_the_readme_leads_with_the_live_links():
    import run_pipeline

    step = next(s for s in run_pipeline.STEPS if s.name == "build_pages")
    readme = open("README.md", encoding="utf-8").read()[:1500]

    assert "build_share_image.py" in step.scripts and {"index.html", "share_image.png"} <= set(step.outputs)
    assert sc.PAGES_BASE in readme and "share_image.png" in readme


def _components():
    base = {"delta": "+0.05", "p": "<0.0001", "n": 1000, "passes": True}
    rows = [("whiff", "Whiff", "0.614", "0.515", True), ("chase", "Chase", "0.533", "0.459", True), ("weak", "Weak contact", "0.288", "0.270", True),
            ("timing", "Timing", "0.631", "0.513", True), ("gb", "Ground ball", "0.446", "0.353", False)]
    comps = [{**base, "key": k, "label": label, "reliability": r, "yoy": y, "in_score": inside} for k, label, r, y, inside in rows]
    comps.append({**base, "key": "composite", "label": "Deception+ (forecast is for next-season whiff rate)", "reliability": "0.741", "yoy": "0.611", "in_score": True})
    return comps


def test_the_piece_dot_plot_sorts_by_steadiness_leaves_out_excluded_pieces_and_ends_with_the_composite():
    svg = sc.piece_dots_svg(_components())

    order = [m for m in ("Timing", "Whiff", "Chase", "Weak contact") if m in svg]
    assert [svg.index(m) for m in order] == sorted(svg.index(m) for m in order)          # 0.631, 0.614, 0.533, 0.288 top to bottom
    assert "Ground ball" not in svg and svg.rindex("Deception+") > svg.index("Weak contact")
    assert "0.631 and 0.513" in svg and svg.startswith('<svg class="dx-dots"') and 'role="img"' in svg


def test_binned_means_split_into_equal_groups_with_intervals_and_drop_a_thin_remainder():
    rows = [(float(x), 100.0 + x / 10 + (1 if x % 2 else -1)) for x in range(1, 201)]

    bins = sc.binned_means(rows, groups=10)

    assert len(bins) == 10 and all(b["n"] == 20 for b in bins)
    assert [b["x"] for b in bins] == sorted(b["x"] for b in bins) and all(b["lo"] < b["y"] < b["hi"] for b in bins)
    assert len(sc.binned_means(rows + [(float(x), 100.0) for x in range(201, 206)], groups=10)) == 9    # 205 rows: nine slices of 21 and a remainder of 16, which is dropped
    assert sc.bin_table_rows(bins).count("<tr>") == 10 and "1 (lowest)" in sc.bin_table_rows(bins) and "10 (highest)" in sc.bin_table_rows(bins)


def test_scatter_points_keep_only_qualified_seasons_with_both_a_score_and_a_stuff_figure():
    point = lambda name, stuff, score, qualified: [name, 2025, stuff, 0, 0, 0, 0, 0, score, 500, False, qualified, 0]

    kept = sc.scatter_points([point("A", 100, 105.0, True), point("B", 100, 105.0, False), point("C", None, 105.0, True), point("D", 100, None, True)])

    assert kept == [["A", 2025, 100, 105.0, 500, False]]


def test_the_on_field_forest_puts_outcomes_that_clear_zero_first_and_mutes_those_that_clear_in_neither_test():
    def outcome(label, lo, gain, own_lo):
        return {"label": label, "deception_over_stuff_location": {"gain": gain, "lo": lo, "hi": gain + 0.02},
                "deception_over_own_stuff_location": {"gain": gain / 2, "lo": own_lo, "hi": gain}}

    validation = {"outcomes": {"walk": outcome("walk rate", -0.02, -0.01, -0.03), "x": outcome("xwOBA allowed", 0.01, 0.04, 0.003),
                               "k": outcome("strikeout rate", 0.03, 0.07, -0.005)}}

    rows = sc.outcome_forest_rows(validation).splitlines()

    assert [r.split("</th>")[0].split(">")[-1] for r in rows] == ["Strikeout rate", "xwOBA allowed", "Walk rate"]      # by gain, brand-style names kept
    assert "dx-muted" in rows[2] and "dx-muted" not in rows[0] and "dx-muted" not in rows[1]
    assert rows[1].count("dx-forest clear") == 2 and rows[0].count("dx-forest clear") == 1


def test_the_dashboard_hero_carries_the_caveat_and_no_figure_or_name_count_is_typed_into_the_text():
    text = open("templates/dashboard.html", encoding="utf-8").read()

    assert "It does not prove deception" in text[:text.index('id="overview"')]                # visible before the first section
    for typed in ("Eight pitchers", "Three seasons", "at least 100 swings", "ten equal", "five groups"):
        assert typed not in text, typed


def test_the_ruler_note_says_which_marked_pitchers_include_100_in_their_interval():
    lore = [{"name": "Rogers, Tyler", "dp_lo": 87.9, "dp_hi": 115.6}, {"name": "Skubal, Tarik", "dp_lo": 136.0, "dp_hi": 154.5},
            {"name": "Hill, Tim", "dp_lo": 76.3, "dp_hi": 103.6}, {"name": "Someone, Else", "dp_lo": 120.0, "dp_hi": 130.0}]

    note = sc.ruler_note(lore)

    assert note == ("Bars show each marked pitcher's 95% interval. Hill and Rogers include 100, so are not clearly different from average. "
                    "Skubal clears it.")                                            # only the named picks, in the ruler's order
    assert sc.ruler_note([{"name": "Skubal, Tarik", "dp_lo": 136.0, "dp_hi": 154.5}]).endswith("Skubal clears it.")
    assert sc.ruler_note([{"name": "Skubal, Tarik", "dp_lo": None, "dp_hi": None}]) == ""


def test_reputation_rows_show_the_95_interval_and_flag_the_ones_that_include_100():
    base = {"name": "Rogers, Tyler", "season": 2025, "n": 530, "deception_plus": 101.7, "dp_pctile": 64, "whiff_index": 67.3, "pctile": 0,
            "gb_index": 129.7, "gb_pctile": 100}

    inside = sc.reputation_rows([{**base, "dp_lo": 87.9, "dp_hi": 115.6}], "lb.html")
    clear = sc.reputation_rows([{**base, "dp_lo": 136.0, "dp_hi": 154.5}], "lb.html")
    missing = sc.reputation_rows([base], "lb.html")

    assert "[87.9, 115.6]" in inside and "includes 100" in inside
    assert "[136.0, 154.5]" in clear and "includes 100" not in clear
    assert missing.count("<td>n/a</td>") == 1 and inside.count("<td>") == 5                 # season, score, interval, whiff and ground ball (the name is a th)


def test_the_scores_csv_has_one_row_per_pitcher_season_without_stuff_plus_and_with_blanks_for_missing_values():
    import csv
    import io

    import build_scores_csv as bsc

    rows = [{"name": "Hill, Tim", "season": 2025, "role": "RP", "qualified": True, "deception_plus": 90.0, "deception_plus_ci_lo": 76.3, "deception_plus_ci_hi": 103.6,
             "proj": None, "proj_lo": None, "proj_hi": None, "proj_basis": None, "stuff_plus": 98.0, "whiff_index": 92.2, "whiff_n": 481.0, "whiff_ci_lo": 80.0, "whiff_ci_hi": 104.0},
            {"name": "Skubal, Tarik", "season": 2025, "role": "SP", "qualified": True, "deception_plus": 145.3, "deception_plus_ci_lo": 136.0, "deception_plus_ci_hi": 154.5,
             "proj": 128.4, "proj_lo": 119.7, "proj_hi": 137.2, "proj_basis": 2.0, "stuff_plus": 120.0, "whiff_index": 138.5, "whiff_n": 1552.0},
            {"name": "Nobody, No", "season": 2024, "role": "RP", "qualified": False, "deception_plus": None}]

    parsed = list(csv.DictReader(io.StringIO(bsc.rows_to_csv(rows))))

    assert [r["player"] for r in parsed] == ["Skubal, Tarik", "Hill, Tim", "Nobody, No"]      # highest score first, unscored last; the comma in a name is quoted
    assert "stuff_plus" not in parsed[0] and "types" not in parsed[0] and len(parsed[0]) == len(bsc.COLUMNS)
    assert parsed[0]["whiff_n"] == "1552" and parsed[0]["projection_seasons_used"] == "2"        # counts are whole numbers
    assert parsed[1]["next_season_projection"] == "" and parsed[2]["deception_plus"] == ""


def test_the_leaderboard_links_the_csv_the_ruler_names_come_from_one_list_and_the_methodology_page_leads_with_what_held_and_what_did_not():
    leaderboard = open("templates/leaderboard.html", encoding="utf-8").read()
    dashboard = open("templates/dashboard.html", encoding="utf-8").read()
    methodology = open("templates/methodology.html", encoding="utf-8").read()

    assert 'href="__CSV_URL__"' in leaderboard
    assert "__RULER_NAMES_JSON__" in dashboard and "__RULER_NOTE__" in dashboard and "'Hill, Tim'" not in dashboard
    assert methodology.index('id="read-first"') < methodology.index('class="dx-toc"')
    assert "What was checked and held" in methodology and "What did not hold, or is limited" in methodology
    import run_pipeline

    step = next(s for s in run_pipeline.STEPS if s.name == "build_pages")
    assert "build_scores_csv.py" in step.scripts and "deception_scores.csv" in step.outputs


def test_the_csv_link_downloads_on_the_standalone_page_and_opens_in_a_new_tab_inside_the_artifact():
    import build_leaderboard

    template = open("templates/leaderboard.html", encoding="utf-8").read()
    source = open(build_leaderboard.__file__, encoding="utf-8").read()

    assert 'href="__CSV_URL__"__CSV_ATTRS__' in template
    assert '" target=\"_blank\" rel=\"noopener\"' not in template and "if args.artifact else \" download\"" in source
