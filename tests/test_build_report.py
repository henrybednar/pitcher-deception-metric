import json

import build_report as br


def test_p_and_q_formats_follow_the_report_conventions():
    assert br.fp(0.00003) == "$<0.0001$***"
    assert br.fp(0.0003) == "$0.0003$***"
    assert br.fp(0.017) == "$0.017$*"
    assert br.fp(0.49) == "$0.49$"
    assert br.fq(0.0019) == "$0.0019$" and br.fq(0.061) == "$0.061$" and br.fq(0.73) == "$0.73$"
    assert br.peq(0.00002) == "p<0.0001" and br.peq(0.0123) == "p=0.012"


def report_with(rows):
    report = br.Report.__new__(br.Report)
    report.ols = {"whiff": {"rows": rows}}
    return report


def test_a_feature_moves_between_tiers_by_its_q_and_p_values():
    rows = [{"feature": "a", "p": 0.0001, "q": 0.001}, {"feature": "b", "p": 0.03, "q": 0.08},
            {"feature": "c", "p": 0.2, "q": 0.4}, {"feature": "d", "p": 0.04, "q": 0.049}]

    survive, nominal, rest = report_with(rows).tiers("whiff")

    assert [r["feature"] for r in survive] == ["a", "d"]
    assert [r["feature"] for r in nominal] == ["b"]
    assert [r["feature"] for r in rest] == ["c"]
