import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "02_machine_learning" / "02_logistic_regression" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("logistic_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_stable_sigmoid_handles_extreme_values() -> None:
    values = np.array([-1000.0, 0.0, 1000.0])
    probabilities = solution.stable_sigmoid(values)
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(1.0)


def test_cross_entropy_from_extreme_logits_is_finite() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    logits = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
    loss = solution.binary_cross_entropy_from_logits(y, logits)
    assert np.isfinite(loss)
    assert loss == pytest.approx(500.0)


def test_analytic_gradients_match_finite_differences_with_l2() -> None:
    X = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 1.0], [-1.0, -2.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    weights = np.array([0.2, -0.3])
    bias = 0.1
    l2 = 0.4
    gradient_weights, gradient_bias = solution.logistic_gradients(
        X, y, weights, bias, l2=l2
    )
    epsilon = 1e-6
    numerical_weights = np.empty_like(weights)
    for index in range(weights.size):
        offset = np.zeros_like(weights)
        offset[index] = epsilon
        plus = solution.logistic_loss(X, y, weights + offset, bias, l2=l2)
        minus = solution.logistic_loss(X, y, weights - offset, bias, l2=l2)
        numerical_weights[index] = (plus - minus) / (2 * epsilon)
    plus_bias = solution.logistic_loss(X, y, weights, bias + epsilon, l2=l2)
    minus_bias = solution.logistic_loss(X, y, weights, bias - epsilon, l2=l2)
    numerical_bias = (plus_bias - minus_bias) / (2 * epsilon)
    np.testing.assert_allclose(gradient_weights, numerical_weights, rtol=1e-5, atol=1e-7)
    assert gradient_bias == pytest.approx(numerical_bias, rel=1e-5)


def test_gradient_descent_learns_separable_data() -> None:
    X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    weights, bias, losses = solution.fit_gradient_descent(
        X, y, learning_rate=0.2, n_steps=1000, l2=0.01
    )
    probabilities = solution.predict_proba(X, weights, bias)
    labels = solution.predict_labels(probabilities)
    np.testing.assert_array_equal(labels, y.astype(int))
    assert losses.shape == (1001,)
    assert losses[-1] < losses[0]


def test_l2_reduces_weight_norm() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    unregularized, _, _ = solution.fit_gradient_descent(
        X, y, learning_rate=0.1, n_steps=500, l2=0.0
    )
    regularized, _, _ = solution.fit_gradient_descent(
        X, y, learning_rate=0.1, n_steps=500, l2=1.0
    )
    assert np.linalg.norm(regularized) < np.linalg.norm(unregularized)


def test_threshold_changes_predictions() -> None:
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    np.testing.assert_array_equal(
        solution.predict_labels(probabilities, 0.5), [0, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        solution.predict_labels(probabilities, 0.3), [0, 1, 1, 1]
    )


def test_training_does_not_modify_initial_weights() -> None:
    X = np.array([[-1.0], [1.0]])
    y = np.array([0.0, 1.0])
    initial = np.array([0.5])
    original = initial.copy()
    solution.fit_gradient_descent(
        X, y, learning_rate=0.1, n_steps=2, initial_weights=initial
    )
    np.testing.assert_array_equal(initial, original)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.logistic_loss(np.ones((3, 1)), np.ones((3, 1)), np.ones(1), 0.0),
        lambda: solution.logistic_loss(np.ones((3, 1)), np.array([0.0, 1.0, 2.0]), np.ones(1), 0.0),
        lambda: solution.logistic_loss(np.ones((3, 1)), np.ones(3), np.ones(1), 0.0, l2=-1.0),
        lambda: solution.predict_labels(np.array([0.2, 1.2])),
        lambda: solution.fit_gradient_descent(
            np.ones((2, 1)), np.array([0.0, 1.0]), learning_rate=0.0, n_steps=10
        ),
    ],
)
def test_invalid_logistic_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()
