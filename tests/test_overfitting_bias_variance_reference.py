import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "02_model_evaluation_selection" / "04_overfitting_bias_variance" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("overfitting_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_polynomial_features_and_exact_cubic_fit() -> None:
    x = np.array([-1.0, 0.0, 2.0])
    features = solution.polynomial_features(x, 3)
    assert features.shape == (3, 4)
    np.testing.assert_allclose(features[2], [1.0, 2.0, 4.0, 8.0])
    y = 1 + 2 * x - 3 * x**2 + 0.5 * x**3
    coefficients = solution.fit_polynomial(x, y, 3)
    np.testing.assert_allclose(solution.predict_polynomial(x, coefficients), y, atol=1e-10)


def test_complexity_curve_exposes_overfitting() -> None:
    x_train = np.linspace(-1.0, 1.0, 8)
    true_train = 0.5 + 2 * x_train - x_train**2 + 0.5 * x_train**3
    noise = np.array([0.12, -0.18, 0.08, -0.05, 0.04, -0.09, 0.16, -0.11])
    y_train = true_train + noise
    x_validation = np.linspace(-0.95, 0.95, 101)
    y_validation = 0.5 + 2 * x_validation - x_validation**2 + 0.5 * x_validation**3
    curve = solution.complexity_curve(
        x_train, y_train, x_validation, y_validation, range(8)
    )
    assert curve["train_mse"][-1] < 1e-20
    assert curve["validation_mse"][-1] > curve["validation_mse"][3]
    assert int(curve["degrees"][np.argmin(curve["validation_mse"])]) < 7


def test_learning_curve_shapes_and_sizes() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.uniform(-1, 1, size=20)
    y_train = np.sin(np.pi * x_train)
    x_validation = np.linspace(-1, 1, 31)
    y_validation = np.sin(np.pi * x_validation)
    result = solution.learning_curve(
        x_train, y_train, x_validation, y_validation, degree=3, train_sizes=[4, 8, 12, 20]
    )
    np.testing.assert_array_equal(result["train_sizes"], [4, 8, 12, 20])
    assert result["train_mse"].shape == (4,)
    assert result["validation_mse"].shape == (4,)
    assert np.all(np.isfinite(result["validation_mse"]))


def test_bias_variance_components_match_hand_calculation() -> None:
    predictions = np.array([[0.0, 2.0], [2.0, 4.0]])
    truth = np.array([1.0, 2.0])
    result = solution.bias_variance_components(predictions, truth, 0.25)
    np.testing.assert_allclose(result["mean_prediction"], [1.0, 3.0])
    np.testing.assert_allclose(result["pointwise_bias_squared"], [0.0, 1.0])
    np.testing.assert_allclose(result["pointwise_variance"], [1.0, 1.0])
    assert result["bias_squared"] == pytest.approx(0.5)
    assert result["variance"] == pytest.approx(1.0)
    assert result["expected_error"] == pytest.approx(1.75)


def test_simulation_is_reproducible_and_high_degree_has_more_variance() -> None:
    x_eval = np.linspace(-0.8, 0.8, 25)
    low_first, truth = solution.simulate_polynomial_predictions(1, 12, 200, x_eval, 0.2, 42)
    low_second, _ = solution.simulate_polynomial_predictions(1, 12, 200, x_eval, 0.2, 42)
    high, _ = solution.simulate_polynomial_predictions(9, 12, 200, x_eval, 0.2, 42)
    np.testing.assert_array_equal(low_first, low_second)
    low_components = solution.bias_variance_components(low_first, truth, 0.04)
    high_components = solution.bias_variance_components(high, truth, 0.04)
    assert high_components["variance"] > low_components["variance"]
    assert low_components["bias_squared"] > 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.polynomial_features(np.array([1.0]), -1),
        lambda: solution.fit_polynomial(np.array([1.0, 2.0]), np.array([1.0]), 1),
        lambda: solution.learning_curve(
            np.arange(5.0), np.arange(5.0), np.arange(2.0), np.arange(2.0), 3, [3]
        ),
        lambda: solution.simulate_polynomial_predictions(5, 5, 10, np.array([0.0]), 0.1, 1),
        lambda: solution.bias_variance_components(np.ones((1, 2)), np.ones(2), 0.1),
    ],
)
def test_invalid_experiment_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()
