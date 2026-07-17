import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = (
    ROOT
    / "watermelon_book"
    / "03_linear_models"
    / "01_linear_regression"
    / "generalized_linear_models"
)
SOLUTION = TOPIC / "reference" / "solution.py"
STARTER = TOPIC / "starter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(SOLUTION, "glm_solution")
starter = load_module(STARTER, "glm_starter")


def exponential_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(5.0).reshape(-1, 1)
    y = np.exp(0.7 * X[:, 0] + 0.2)
    return X, y


def test_linear_predictor_has_one_value_per_sample() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    eta = solution.linear_predictor(X, np.array([0.5, -1.0]), 2.0)
    assert eta.shape == (2,)
    np.testing.assert_array_equal(eta, [0.5, -0.5])


def test_identity_link_is_ordinary_linear_regression_special_case() -> None:
    values = np.array([-2.0, 0.0, 3.0])
    linked = solution.apply_link(values, link="identity")
    restored = solution.inverse_link(linked, link="identity")
    np.testing.assert_array_equal(linked, values)
    np.testing.assert_array_equal(restored, values)
    assert not np.shares_memory(linked, values)


def test_log_link_and_inverse_round_trip_positive_values() -> None:
    values = np.array([0.1, 1.0, 10.0, 100.0])
    linked = solution.apply_link(values, link="log")
    restored = solution.inverse_link(linked, link="log")
    np.testing.assert_allclose(restored, values, rtol=1e-14)
    assert np.all(restored > 0)


def test_logit_link_and_stable_inverse_round_trip_probabilities() -> None:
    probabilities = np.array([1e-12, 0.1, 0.5, 0.9, 1 - 1e-12])
    logits = solution.apply_link(probabilities, link="logit")
    restored = solution.inverse_link(logits, link="logit")
    np.testing.assert_allclose(restored, probabilities, rtol=1e-12, atol=1e-15)


def test_logit_inverse_is_finite_for_extreme_linear_predictors() -> None:
    eta = np.array([-1000.0, 0.0, 1000.0])
    probabilities = solution.inverse_link(eta, link="logit")
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_array_equal(probabilities, [0.0, 0.5, 1.0])


def test_predict_mean_applies_inverse_link_after_linear_predictor() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    weights = np.array([0.7])
    prediction = solution.predict_mean(X, weights, 0.2, link="log")
    np.testing.assert_allclose(prediction, np.exp([0.2, 0.9, 1.6]))
    assert prediction.shape == (3,) and np.all(prediction > 0)


def test_log_linear_fit_recovers_exact_exponential_relation() -> None:
    X, y = exponential_data()
    weights, bias = solution.fit_log_linear(X, y)
    prediction = solution.predict_log_linear(X, weights, bias)
    np.testing.assert_allclose(weights, [0.7], atol=1e-12)
    assert bias == pytest.approx(0.2, abs=1e-12)
    np.testing.assert_allclose(prediction, y, rtol=1e-12)
    assert solution.mean_squared_link_error(y, prediction, link="log") < 1e-25


def test_multiplying_target_changes_log_intercept_not_slope() -> None:
    X, y = exponential_data()
    weights, bias = solution.fit_log_linear(X, y)
    scaled_weights, scaled_bias = solution.fit_log_linear(X, 10.0 * y)
    np.testing.assert_allclose(scaled_weights, weights, atol=1e-12)
    assert scaled_bias == pytest.approx(bias + np.log(10.0), abs=1e-12)


def test_log_linear_fit_without_intercept_recovers_zero_intercept_relation() -> None:
    X = np.arange(1.0, 6.0).reshape(-1, 1)
    y = np.exp(0.4 * X[:, 0])
    weights, bias = solution.fit_log_linear(X, y, fit_intercept=False)
    np.testing.assert_allclose(weights, [0.4], atol=1e-12)
    assert bias == 0.0


def test_log_linear_fit_handles_rank_deficient_design() -> None:
    first = np.arange(1.0, 5.0)
    X = np.column_stack((first, 2.0 * first))
    y = np.exp(0.3 * first + 0.1)
    weights, bias = solution.fit_log_linear(X, y)
    prediction = solution.predict_log_linear(X, weights, bias)
    assert np.all(np.isfinite(weights)) and np.isfinite(bias)
    np.testing.assert_allclose(prediction, y, rtol=1e-12)


def test_link_scale_error_is_not_the_same_as_original_scale_error() -> None:
    y_true = np.array([1.0, 10.0, 100.0])
    y_pred = np.array([2.0, 20.0, 200.0])
    identity_error = solution.mean_squared_link_error(
        y_true, y_pred, link="identity"
    )
    log_error = solution.mean_squared_link_error(y_true, y_pred, link="log")
    assert identity_error == pytest.approx((1 + 100 + 10000) / 3)
    assert log_error == pytest.approx(np.log(2.0) ** 2)
    assert identity_error != pytest.approx(log_error)


def test_fitting_and_prediction_do_not_modify_inputs() -> None:
    X, y = exponential_data()
    original_X, original_y = X.copy(), y.copy()
    weights, bias = solution.fit_log_linear(X, y)
    original_weights = weights.copy()
    solution.predict_log_linear(X, weights, bias)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize(
    "values, link, message",
    [
        (np.array([0.0, 1.0]), "log", "严格大于0"),
        (np.array([-1.0, 1.0]), "log", "严格大于0"),
        (np.array([0.0, 0.5]), "logit", "位于"),
        (np.array([0.5, 1.0]), "logit", "位于"),
        (np.array([0.5]), "unknown", "link"),
    ],
)
def test_invalid_link_domains_and_names_are_rejected(values, link, message) -> None:
    with pytest.raises(ValueError, match=message):
        solution.apply_link(values, link=link)


def test_log_inverse_overflow_is_reported() -> None:
    with pytest.raises(FloatingPointError, match="溢出"):
        solution.inverse_link(np.array([1000.0]), link="log")


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.linear_predictor(np.ones(3), np.ones(1), 0.0),
        lambda: solution.linear_predictor(np.ones((3, 2)), np.ones(3), 0.0),
        lambda: solution.linear_predictor(np.ones((3, 1)), np.ones(1), True),
        lambda: solution.fit_log_linear(np.ones((2, 1)), np.array([1.0, 0.0])),
        lambda: solution.fit_log_linear(
            np.ones((2, 1)), np.ones((2, 1))
        ),
        lambda: solution.fit_log_linear(
            np.ones((2, 1)), np.ones(2), fit_intercept=1
        ),
        lambda: solution.mean_squared_link_error(
            np.ones(2), np.ones((2, 1)), link="identity"
        ),
    ],
)
def test_invalid_shapes_parameters_and_positive_targets_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()


@pytest.mark.parametrize(
    "function_name",
    [
        "linear_predictor",
        "apply_link",
        "inverse_link",
        "predict_mean",
        "fit_log_linear",
        "predict_log_linear",
        "mean_squared_link_error",
    ],
)
def test_student_entry_points_remain_unimplemented(function_name: str) -> None:
    function = getattr(starter, function_name)
    X, y = exponential_data()
    with pytest.raises(NotImplementedError):
        if function_name == "linear_predictor":
            function(X, np.ones(1), 0.0)
        elif function_name in {"apply_link", "inverse_link"}:
            function(y, link="identity")
        elif function_name == "predict_mean":
            function(X, np.ones(1), 0.0, link="identity")
        elif function_name == "fit_log_linear":
            function(X, y)
        elif function_name == "predict_log_linear":
            function(X, np.ones(1), 0.0)
        else:
            function(y, y, link="identity")


def test_guided_demo_runs_and_shows_link_round_trip() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "X / y shapes: (5, 1) (5,)" in result.stdout
    assert "fitted weights / bias: [0.7] 0.2" in result.stdout
    assert "eta shape: (5,)" in result.stdout
    assert "logit round trip: [0.1 0.5 0.9]" in result.stdout
