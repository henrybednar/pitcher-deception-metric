from pathlib import Path

import pytest

import check_readme_claims as claims


def outputs():
    return {
        "text": {"COMPOSITE_R": "0.740", "YOY_R": "0.615", "YOY_QUAL_R": "0.687", "YOY_QUAL_N": "580", "PROJ_COVER": "81%"},
        "projection": {"coefficients": {"one_season": {"intercept": 39.92, "this": 0.593},
                                        "two_seasons": {"intercept": 31.34, "this": 0.488, "previous": 0.187}},
                       "backtest": {"rmse_projection": 7.03, "rmse_raw_score": 8.0, "rmse_league_average": 8.99}},
        "drivers": {c: {"features": [{"q": 0.01}, {"q": 0.2}]} for c in claims.COMPONENT_ORDER},
    }


README = ("Composite reliability (split-half) is 0.740 and year-over-year correlation is 0.615 across every scored pair of seasons, "
          "and 0.687 among the 580 pairs where the pitcher qualified in both seasons. "
          "It covers 81% of outcomes. (1 of 2 for whiff, 1 for chase, 1 for ground ball, 1 for weak contact, 1 for timing, 1 for whiff miss)")


def pick(*names):
    return [c for c in claims.CLAIMS if c.name in names]


def test_matching_claims_pass():
    names = ("composite reliability and year-over-year", "year-over-year among qualified pairs", "projection coverage", "driver survivors")

    assert claims.check(README, outputs(), pick(*names)) == []


def test_a_stale_number_is_reported_with_both_values():
    stale = README.replace("is 0.740 and", "is 0.742 and")

    problems = claims.check(stale, outputs(), pick("composite reliability and year-over-year"))

    assert len(problems) == 1 and "['0.742', '0.615']" in problems[0] and "['0.740', '0.615']" in problems[0]


def test_a_reworded_sentence_is_reported_instead_of_silently_skipped():
    reworded = README.replace("year-over-year correlation is", "stability from one year to the next is")

    problems = claims.check(reworded, outputs(), pick("composite reliability and year-over-year"))

    assert len(problems) == 1 and "0 matches" in problems[0]


def test_a_sentence_that_appears_twice_is_reported():
    problems = claims.check(README + " " + README, outputs(), pick("projection coverage"))

    assert len(problems) == 1 and "2 matches" in problems[0]


def test_a_missing_output_file_is_named_not_crashed_on():
    problems = claims.check(README, {"text": {}}, pick("projection coefficients"))

    assert len(problems) == 1 and "not available" in problems[0]


def test_projection_coefficients_round_to_the_readmes_precision():
    assert claims.projection_coefficients(outputs()["projection"]) == ["39.9", "0.59", "31.3", "0.49", "0.19"]


def test_survivors_count_features_under_the_correction_in_the_readmes_order():
    drivers = {c: {"features": [{"q": 0.01}] * i + [{"q": 0.5}]} for i, c in enumerate(claims.COMPONENT_ORDER)}

    assert claims.survivors(drivers) == ["0", "1", "1", "2", "3", "4", "5"]         # whiff survivors, features tested, then the other five outcomes


NEEDED = ["site_stats.json", "outcome_validation.json", "projection.json", "driver_analysis.json", "membership_check.json"]


@pytest.mark.skipif(not all(Path("output", name).exists() for name in NEEDED) or not Path("README.md").exists(), reason="needs a pipeline run")
def test_the_readme_matches_the_outputs_of_the_last_run():
    problems = claims.check(Path("README.md").read_text(encoding="utf-8"), claims.load_outputs())

    assert problems == []


def test_membership_sentence_takes_absolute_differences_and_signed_intervals():
    m = {"+calledstrike": {"diff": -0.039, "diff_lo": -0.066, "diff_hi": -0.014}, "+align": {"diff": -0.029, "diff_lo": -0.048, "diff_hi": -0.010},
         "-whiffmiss": {"diff": -0.048, "diff_lo": -0.070, "diff_hi": -0.028}, "-whiff": {"diff": -0.051, "diff_lo": -0.079, "diff_hi": -0.025}}

    assert claims.membership(m) == ["0.039", "-0.066", "-0.014", "0.029", "-0.048", "-0.010", "0.048", "-0.070", "-0.028", "0.051", "-0.079", "-0.025"]


def test_stuff_correlation_text_is_split_into_the_readmes_figures():
    text = {"STUFF_R_BY_SEASON": "0.315 in 2024, 0.439 in 2025, and 0.383 in 2026",
            "STUFF_R_MEMBERS": "whiff .302, chase .220, weak contact .214, timing .130, and whiff miss distance .358"}

    assert claims.stuff_by_season(text) == ["0.315", "0.439", "0.383"]
    assert claims.stuff_by_member(text) == ["0.30", "0.22", "0.21", "0.13", "0.36"]


@pytest.mark.parametrize("claim", claims.CLAIMS, ids=lambda c: c.name)
def test_every_claim_pattern_matches_exactly_one_sentence_of_the_readme(claim):
    import re

    assert len(re.findall(claim.pattern, Path("README.md").read_text(encoding="utf-8"))) == 1


def test_main_writes_a_report_and_fails_when_a_claim_is_stale(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    (tmp_path / "README.md").write_text("Composite reliability is 0.742 and year-over-year correlation is 0.615 across every scored pair of seasons.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(claims, "load_outputs", lambda: outputs())
    monkeypatch.setattr(claims, "CLAIMS", pick("composite reliability and year-over-year"))

    code = claims.main()

    report = (tmp_path / "output" / "readme_claims_check.txt").read_text(encoding="utf-8")
    assert code == 1 and "MISMATCH composite reliability and year-over-year" in report and "0 of 1 README claims match" in report


def test_fix_replaces_only_the_figures_of_a_drifted_claim_and_leaves_the_words():
    stale = README.replace("is 0.740 and", "is 0.742 and").replace("(1 of 2 for whiff", "(3 of 2 for whiff")

    fixed, changed = claims.fix(stale, outputs(), pick("composite reliability and year-over-year", "driver survivors"))

    assert "is 0.740 and year-over-year correlation is 0.615 across every scored pair of seasons" in fixed
    assert changed == ["composite reliability and year-over-year", "driver survivors"]
    assert fixed.count("0.742") == 0 and "(1 of 2 for whiff, 1 for chase" in fixed


def test_fix_does_nothing_to_a_reworded_or_duplicated_claim():
    reworded = README.replace("year-over-year correlation is", "stability is")
    twice = README + " " + README.replace("0.740", "0.741")

    assert claims.fix(reworded, outputs(), pick("composite reliability and year-over-year")) == (reworded, [])
    assert claims.fix(twice, outputs(), pick("composite reliability and year-over-year")) == (twice, [])


def test_validation_ranges_come_from_the_pitch_types_in_model_validation():
    validation = {"whiff": {"by_pitch_type": [{"auc": 0.76}, {"auc": 0.851}], "overall": {"logloss": 0.4185, "baseline_logloss": 0.5258}},
                  "timing": {"by_pitch_type": [{"r2": 0.065}, {"r2": 0.231}]}}

    assert claims.metric_range(validation, "whiff", "auc") == ["0.76", "0.85"]
    assert claims.metric_range(validation, "timing", "r2") == ["0.07", "0.23"]
