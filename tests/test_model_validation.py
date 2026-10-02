import numpy as np
import pandas as pd
import pytest

import model_validation as mv


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_constant_log_loss_of_an_even_split_is_ln_two():
    assert mv.constant_log_loss(np.array([0, 1, 0, 1])) == pytest.approx(np.log(2))


def test_a_constant_prediction_has_chance_discrimination_and_no_log_loss_gain():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.3).astype(float)

    metrics = mv.binary_metrics(y, np.full(5000, y.mean()))

    assert metrics["auc"] == pytest.approx(0.5)
    assert metrics["logloss"] == pytest.approx(metrics["baseline_logloss"], abs=1e-4)


def test_recalibration_slope_is_one_when_predictions_are_calibrated_and_below_one_when_overconfident():
    rng = np.random.default_rng(1)
    z = rng.normal(size=60_000)
    y = (rng.random(60_000) < sigmoid(z)).astype(float)

    calibrated = mv.recalibration_slope(y, sigmoid(z), binary=True)
    overconfident = mv.recalibration_slope(y, sigmoid(2 * z), binary=True)

    assert calibrated == pytest.approx(1.0, abs=0.05)
    assert overconfident == pytest.approx(0.5, abs=0.06)


def test_recalibration_slope_for_a_continuous_outcome_is_the_fitted_line_slope():
    rng = np.random.default_rng(2)
    p = rng.normal(size=20_000)

    assert mv.recalibration_slope(2.0 * p + rng.normal(scale=0.1, size=20_000), p, binary=False) == pytest.approx(2.0, abs=0.02)


def test_calibration_bins_split_evenly_in_predicted_order_and_account_for_every_row():
    rng = np.random.default_rng(3)
    p = rng.random(1000)

    bins = mv.calibration_bins((rng.random(1000) < p).astype(float), p)

    assert len(bins) == mv.N_BINS
    assert [b["pred"] for b in bins] == sorted(b["pred"] for b in bins)
    assert sum(b["n"] for b in bins) == 1000


def test_regression_metrics_report_r2_and_the_constant_guess_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = mv.regression_metrics(y, y)

    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["baseline_rmse"] == pytest.approx(np.std(y))


def test_validate_outcome_skips_thin_pitch_types_and_orders_the_rest_by_size():
    rng = np.random.default_rng(4)
    frames = []
    for pitch_type, n in (("SL", 2000), ("FF", 3000), ("CU", 500)):
        p = rng.uniform(0.1, 0.5, n)
        frames.append(pd.DataFrame({"pitch_type": pitch_type, "is_swing": True,
                                    "is_whiff": (rng.random(n) < p).astype(int), "whiff_expected_full": p}))

    result = mv.validate_outcome(pd.concat(frames, ignore_index=True), "whiff")

    assert [r["pitch_type"] for r in result["by_pitch_type"]] == ["FF", "SL"]
    assert result["kind"] == "classify" and result["in_score"] is True and result["n"] == 5500
    assert len(result["calibration"]["bins"]) == mv.N_BINS
    assert result["overall"]["auc"] > 0.55


def test_validation_text_quotes_the_spread_across_pitch_types_and_the_calibration_slopes():
    from export_site_stats import validation_text

    def outcome(key, metric, values, slope):
        return key, {"by_pitch_type": [{metric: v} for v in values], "calibration": {"slope": slope}}

    validation = {"outcomes": dict([
        outcome("gb", "auc", [0.616, 0.673], 0.92), outcome("calledstrike", "auc", [0.984, 0.991], 1.01),
        outcome("timing", "r2", [0.054, 0.206], 0.998), outcome("whiffmiss", "r2", [0.061, 0.395], 0.995),
        outcome("whiff", "auc", [0.75, 0.84], 0.998), outcome("chase", "auc", [0.81, 0.88], 1.021),
        outcome("weak", "auc", [0.68, 0.75], 0.975)])}

    text = validation_text(validation)

    assert text["VAL_AUC_GB"] == "0.62 to 0.67"
    assert text["VAL_R2_TIMING"] == "0.05 to 0.21"
    assert text["VAL_SLOPE_MEMBERS"] == "0.97 to 1.02"
    assert text["VAL_SLOPE_GB"] == "0.92"
