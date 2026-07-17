import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "03_backpropagation"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("network_backprop_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def sample_problem() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    X = np.array([[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8]])
    y = np.array([[0.0], [1.0], [0.0]])
    parameters = {
        "W1": np.array([[0.1, -0.2], [0.3, 0.4]]),
        "b1": np.array([0.05, -0.05]),
        "W2": np.array([[0.2], [-0.3]]),
        "b2": np.array([0.1]),
    }
    return X, y, parameters


def test_backward_gradient_shapes_match_parameters() -> None:
    X, y, parameters = sample_problem()
    gradients = solution.backward_pass(parameters, solution.forward_pass(X, parameters), y)
    assert set(gradients) == set(parameters)
    for key in parameters:
        assert gradients[key].shape == parameters[key].shape
        assert np.all(np.isfinite(gradients[key]))


def test_output_layer_gradient_matches_direct_formula() -> None:
    X, y, parameters = sample_problem()
    cache = solution.forward_pass(X, parameters)
    gradients = solution.backward_pass(parameters, cache, y)
    dz2 = (cache["probabilities"] - y) / len(X)
    np.testing.assert_allclose(gradients["W2"], cache["a1"].T @ dz2)
    np.testing.assert_allclose(gradients["b2"], dz2.sum(axis=0))


def test_hidden_layer_gradient_matches_direct_formula() -> None:
    X, y, parameters = sample_problem()
    cache = solution.forward_pass(X, parameters)
    gradients = solution.backward_pass(parameters, cache, y)
    dz2 = (cache["probabilities"] - y) / len(X)
    da1 = dz2 @ parameters["W2"].T
    dz1 = da1 * cache["a1"] * (1.0 - cache["a1"])
    np.testing.assert_allclose(gradients["W1"], X.T @ dz1)
    np.testing.assert_allclose(gradients["b1"], dz1.sum(axis=0))


def test_all_analytic_gradients_match_centered_differences() -> None:
    X, y, parameters = sample_problem()
    analytic = solution.backward_pass(parameters, solution.forward_pass(X, parameters), y)
    numerical = solution.finite_difference_gradients(X, y, parameters, epsilon=1e-5)
    assert solution.maximum_relative_error(analytic, numerical) < 1e-6


def test_each_parameter_gradient_matches_numerically() -> None:
    X, y, parameters = sample_problem()
    analytic = solution.backward_pass(parameters, solution.forward_pass(X, parameters), y)
    numerical = solution.finite_difference_gradients(X, y, parameters)
    for key in parameters:
        np.testing.assert_allclose(analytic[key], numerical[key], rtol=1e-6, atol=1e-8)


def test_single_sample_and_three_hidden_units() -> None:
    X = np.array([[0.5, -1.0]])
    y = np.array([[1.0]])
    parameters = {
        "W1": np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.4]]),
        "b1": np.zeros(3),
        "W2": np.array([[0.2], [-0.3], [0.1]]),
        "b2": np.zeros(1),
    }
    cache = solution.forward_pass(X, parameters)
    gradients = solution.backward_pass(parameters, cache, y)
    assert cache["probabilities"].shape == (1, 1)
    assert gradients["W1"].shape == (2, 3)
    assert gradients["W2"].shape == (3, 1)


def test_extreme_logits_keep_loss_and_gradients_finite() -> None:
    X = np.array([[-1.0], [1.0]])
    y = np.array([[0.0], [1.0]])
    parameters = {
        "W1": np.array([[1000.0]]),
        "b1": np.zeros(1),
        "W2": np.array([[1000.0]]),
        "b2": np.array([-500.0]),
    }
    cache = solution.forward_pass(X, parameters)
    loss = solution.loss_for_parameters(X, y, parameters)
    gradients = solution.backward_pass(parameters, cache, y)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(value)) for value in gradients.values())


def test_backward_and_numeric_checks_do_not_modify_inputs() -> None:
    X, y, parameters = sample_problem()
    original_X = X.copy()
    original_y = y.copy()
    original_parameters = {key: value.copy() for key, value in parameters.items()}
    cache = solution.forward_pass(X, parameters)
    solution.backward_pass(parameters, cache, y)
    solution.finite_difference_gradients(X, y, parameters)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    for key in parameters:
        np.testing.assert_array_equal(parameters[key], original_parameters[key])


def test_wrong_cache_shape_or_value_is_rejected() -> None:
    X, y, parameters = sample_problem()
    missing = solution.forward_pass(X, parameters)
    missing.pop("z1")
    with pytest.raises(ValueError):
        solution.backward_pass(parameters, missing, y)

    changed = solution.forward_pass(X, parameters)
    changed["a1"][0, 0] += 0.1
    with pytest.raises(ValueError):
        solution.backward_pass(parameters, changed, y)


@pytest.mark.parametrize(
    "y",
    [
        [0.0, 1.0, 0.0],
        np.array(1.0),
        np.array([0.0, 1.0, 0.0]),
        np.array([[0.0], [1.0]]),
        np.array([[0.0], [2.0], [0.0]]),
        np.array([[0.0], [1.0], [np.nan]]),
    ],
)
def test_invalid_labels_are_rejected(y) -> None:
    X, _, parameters = sample_problem()
    with pytest.raises(ValueError):
        solution.loss_for_parameters(X, y, parameters)


def test_wrong_parameter_shape_is_rejected() -> None:
    X, y, parameters = sample_problem()
    parameters["W2"] = np.ones(2)
    with pytest.raises(ValueError):
        solution.loss_for_parameters(X, y, parameters)


@pytest.mark.parametrize("epsilon", [0.0, -1e-5, np.nan, True])
def test_invalid_finite_difference_steps_are_rejected(epsilon) -> None:
    X, y, parameters = sample_problem()
    with pytest.raises(ValueError):
        solution.finite_difference_gradients(X, y, parameters, epsilon=epsilon)


def test_relative_error_rejects_wrong_keys_shapes_and_nonfinite_values() -> None:
    X, y, parameters = sample_problem()
    gradients = solution.backward_pass(parameters, solution.forward_pass(X, parameters), y)

    missing = {key: value.copy() for key, value in gradients.items() if key != "b2"}
    with pytest.raises(ValueError):
        solution.maximum_relative_error(gradients, missing)

    wrong_shape = {key: value.copy() for key, value in gradients.items()}
    wrong_shape["b2"] = np.zeros(2)
    with pytest.raises(ValueError):
        solution.maximum_relative_error(gradients, wrong_shape)

    nonfinite = {key: value.copy() for key, value in gradients.items()}
    nonfinite["b2"][0] = np.nan
    with pytest.raises(ValueError):
        solution.maximum_relative_error(gradients, nonfinite)


def test_guided_demo_reports_every_gradient_shape() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "probabilities / y: (3, 1) (3, 1)" in result.stdout
    assert "dz2: (3, 1)" in result.stdout
    assert "dW2 / db2: (2, 1) (1,)" in result.stdout
    assert "da1 / dz1: (3, 2) (3, 2)" in result.stdout
    assert "dW1 / db1: (2, 2) (2,)" in result.stdout
