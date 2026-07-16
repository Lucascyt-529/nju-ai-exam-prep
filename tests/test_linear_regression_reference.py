import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "03_linear_models" / "01_linear_regression" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("linear_regression_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_prediction_loss_and_shapes() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])
    prediction = solution.predict(X, np.array([2.0]), 1.0)
    np.testing.assert_allclose(prediction, y)
    assert prediction.shape == (3,)
    assert solution.mean_squared_error(y, prediction) == 0.0


def test_analytic_gradients_match_finite_differences() -> None:
    X = np.array([[1.0, 2.0], [2.0, -1.0], [3.0, 0.5]])
    y = np.array([1.0, 2.0, 4.0])
    w = np.array([0.3, -0.2])
    b = 0.4
    gradient_w, gradient_b = solution.mse_gradients(X, y, w, b)
    epsilon = 1e-6

    numerical_w = np.empty_like(w)
    for index in range(w.size):
        offset = np.zeros_like(w)
        offset[index] = epsilon
        plus = solution.mean_squared_error(y, solution.predict(X, w + offset, b))
        minus = solution.mean_squared_error(y, solution.predict(X, w - offset, b))
        numerical_w[index] = (plus - minus) / (2 * epsilon)
    plus_b = solution.mean_squared_error(y, solution.predict(X, w, b + epsilon))
    minus_b = solution.mean_squared_error(y, solution.predict(X, w, b - epsilon))
    numerical_b = (plus_b - minus_b) / (2 * epsilon)

    np.testing.assert_allclose(gradient_w, numerical_w, rtol=1e-5, atol=1e-7)
    assert gradient_b == pytest.approx(numerical_b, rel=1e-5)


def test_least_squares_recovers_exact_line() -> None:
    X = np.arange(1, 6, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    w, b = solution.fit_least_squares(X, y)
    np.testing.assert_allclose(w, [2.0], atol=1e-12)
    assert b == pytest.approx(1.0, abs=1e-12)


def test_least_squares_handles_rank_deficiency() -> None:
    first = np.arange(1, 5, dtype=float)
    X = np.column_stack((first, 2 * first))
    y = 3 * first + 1
    w, b = solution.fit_least_squares(X, y)
    predictions = solution.predict(X, w, b)
    np.testing.assert_allclose(predictions, y, atol=1e-10)
    assert np.all(np.isfinite(w)) and np.isfinite(b)


def test_gradient_descent_converges_to_exact_line() -> None:
    X = np.arange(1, 6, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    w, b, losses = solution.fit_gradient_descent(
        X, y, learning_rate=0.03, n_steps=3000
    )
    np.testing.assert_allclose(w, [2.0], atol=1e-5)
    assert b == pytest.approx(1.0, abs=3e-5)
    assert losses.shape == (3001,)
    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-9


def test_gradient_descent_does_not_modify_initial_weights() -> None:
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    initial = np.array([0.5])
    original = initial.copy()
    solution.fit_gradient_descent(
        X, y, learning_rate=0.01, n_steps=5, initial_w=initial
    )
    np.testing.assert_array_equal(initial, original)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.fit_least_squares(np.ones((3, 1)), np.ones((3, 1))),
        lambda: solution.predict(np.ones((3, 2)), np.ones(3), 0.0),
        lambda: solution.fit_gradient_descent(
            np.ones((3, 1)), np.ones(3), learning_rate=0.0, n_steps=10
        ),
        lambda: solution.fit_gradient_descent(
            np.ones((3, 1)), np.ones(3), learning_rate=0.1, n_steps=0
        ),
    ],
)
def test_invalid_linear_regression_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()
