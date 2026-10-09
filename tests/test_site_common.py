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
    assert "<td><0.0001</td>" not in html_rows and "<td>&lt;0.0001</td>" in html_rows


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

    assert set(standalone) == set(artifact) == {"dashboard", "leaderboard", "methodology"}
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


def test_a_gain_cell_shades_only_when_the_interval_is_above_zero_and_stacks_the_interval_under_the_gain():
    clear = sc.gain_cell({"gain": 0.071, "lo": 0.032, "hi": 0.110})
    unclear = sc.gain_cell({"gain": -0.002, "lo": -0.010, "hi": 0.004})

    assert clear.startswith('<td class="dx-pass">') and "+0.071" in clear and "[+0.032, +0.110]" in clear
    assert unclear.startswith('<td class="">') and "-0.002" in unclear and "[-0.010, +0.004]" in unclear


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

    for template in ("templates/dashboard.html", "templates/methodology.html"):
        text = open(template, encoding="utf-8").read()
        headings = re.findall(r"<(h[1-6])\b", text)

        assert headings[0] == "h1" and headings.count("h1") == 1, template
        assert "<main " in text and "</main>" in text and "<footer " in text and 'href="#main"' in text, template
