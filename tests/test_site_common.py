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
