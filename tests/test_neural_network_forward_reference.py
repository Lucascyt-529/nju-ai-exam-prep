import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "02_forward_propagation"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("network_forward_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def simple_parameters() -> dict[str, np.ndarray]:
    return {
        "W1": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "b1": np.zeros(2),
        "W2": np.array([[1.0], [-1.0]]),
        "b2": np.zeros(1),
    }


def test_stable_sigmoid_handles_extreme_values() -> None:
    values = np.array([-1000.0, 0.0, 1000.0])
    with np.errstate(over="raise"):
        result = solution.stable_sigmoid(values)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.0)
    assert np.all(np.diff(result) > 0)


def test_sigmoid_preserves_shape_and_input() -> None:
    values = np.array([[-2.0, 0.0], [1.0, 3.0]])
    original = values.copy()
    result = solution.stable_sigmoid(values)
    assert result.shape == values.shape
    np.testing.assert_array_equal(values, original)


def test_initialization_shapes_and_zero_biases() -> None:
    parameters = solution.initialize_parameters(3, 4, seed=7)
    assert parameters["W1"].shape == (3, 4)
    assert parameters["b1"].shape == (4,)
    assert parameters["W2"].shape == (4, 1)
    assert parameters["b2"].shape == (1,)
    np.testing.assert_array_equal(parameters["b1"], np.zeros(4))
    np.testing.assert_array_equal(parameters["b2"], np.zeros(1))


def test_initialization_is_reproducible_but_seed_sensitive() -> None:
    first = solution.initialize_parameters(2, 3, seed=11)
    second = solution.initialize_parameters(2, 3, seed=11)
    different = solution.initialize_parameters(2, 3, seed=12)
    for key in solution.PARAMETER_KEYS:
        np.testing.assert_array_equal(first[key], second[key])
    assert not np.array_equal(first["W1"], different["W1"])


def test_forward_pass_values_and_cache_shapes() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    parameters = simple_parameters()
    cache = solution.forward_pass(X, parameters)
    expected_hidden = 1.0 / (1.0 + np.exp(-(X @ parameters["W1"])))
    expected_logits = expected_hidden @ parameters["W2"]
    expected_probabilities = 1.0 / (1.0 + np.exp(-expected_logits))
    np.testing.assert_allclose(cache["a1"], expected_hidden)
    np.testing.assert_allclose(cache["z2"], expected_logits)
    np.testing.assert_allclose(cache["probabilities"], expected_probabilities)
    assert cache["X"].shape == (3, 2)
    assert cache["z1"].shape == cache["a1"].shape == (3, 2)
    assert cache["z2"].shape == cache["probabilities"].shape == (3, 1)


def test_forward_pass_does_not_modify_or_alias_input() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    original = X.copy()
    cache = solution.forward_pass(X, simple_parameters())
    np.testing.assert_array_equal(X, original)
    cache["X"][0, 0] = 99.0
    np.testing.assert_array_equal(X, original)


def test_label_column_conversion_is_explicit_and_independent() -> None:
    y = np.array([0, 1, 1])
    column = solution.as_column_labels(y)
    assert column.shape == (3, 1)
    assert column.dtype == float
    column[0, 0] = 1.0
    np.testing.assert_array_equal(y, [0, 1, 1])


def test_explicit_column_avoids_n_by_n_broadcasting() -> None:
    probabilities = np.array([[0.2], [0.7], [0.9]])
    y_vector = np.array([0.0, 1.0, 1.0])
    wrong_error = probabilities - y_vector
    correct_error = probabilities - solution.as_column_labels(y_vector)
    assert wrong_error.shape == (3, 3)
    assert correct_error.shape == (3, 1)


def test_zero_logits_have_log_two_loss() -> None:
    y = np.array([[0.0], [1.0], [1.0]])
    logits = np.zeros((3, 1))
    assert solution.binary_cross_entropy_from_logits(y, logits) == pytest.approx(np.log(2.0))


def test_extreme_logits_have_finite_correct_and_wrong_losses() -> None:
    correct_y = np.array([[0.0], [1.0]])
    correct_logits = np.array([[-1000.0], [1000.0]])
    wrong_y = 1.0 - correct_y
    correct_loss = solution.binary_cross_entropy_from_logits(correct_y, correct_logits)
    wrong_loss = solution.binary_cross_entropy_from_logits(wrong_y, correct_logits)
    assert np.isfinite(correct_loss) and correct_loss == pytest.approx(0.0)
    assert np.isfinite(wrong_loss) and wrong_loss == pytest.approx(1000.0)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0],
        np.array([]),
        np.array([1.0, np.nan]),
        np.array(["a"]),
    ],
)
def test_invalid_sigmoid_inputs_are_rejected(values) -> None:
    with pytest.raises(ValueError):
        solution.stable_sigmoid(values)


@pytest.mark.parametrize(
    "args",
    [(0, 2, 0), (2, 0, 0), (True, 2, 0), (2, 2, True), (2, 2, 1.5)],
)
def test_invalid_initialization_arguments_are_rejected(args) -> None:
    n_features, n_hidden, seed = args
    with pytest.raises(ValueError):
        solution.initialize_parameters(n_features, n_hidden, seed=seed)


@pytest.mark.parametrize(
    "X",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[1.0, np.inf]]),
        np.array([["a", "b"]]),
    ],
)
def test_invalid_forward_feature_arrays_are_rejected(X: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solution.forward_pass(X, simple_parameters())


def test_wrong_parameter_keys_shapes_and_values_are_rejected() -> None:
    X = np.ones((2, 2))
    missing = simple_parameters()
    missing.pop("b2")
    with pytest.raises(ValueError):
        solution.forward_pass(X, missing)

    wrong_shape = simple_parameters()
    wrong_shape["W2"] = np.ones((2,))
    with pytest.raises(ValueError):
        solution.forward_pass(X, wrong_shape)

    nonfinite = simple_parameters()
    nonfinite["b1"] = np.array([0.0, np.nan])
    with pytest.raises(ValueError):
        solution.forward_pass(X, nonfinite)


@pytest.mark.parametrize(
    "y",
    [
        np.array([[0], [1]]),
        np.array([]),
        np.array([0, 2]),
        np.array([0.0, np.nan]),
        np.array(["0", "1"]),
    ],
)
def test_invalid_label_vectors_are_rejected(y: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solution.as_column_labels(y)


@pytest.mark.parametrize(
    "y, logits",
    [
        (np.array([0.0, 1.0]), np.zeros((2, 1))),
        (np.array([[0.0], [1.0]]), np.zeros((2,))),
        (np.array([[0.0], [2.0]]), np.zeros((2, 1))),
        (np.array([[0.0], [1.0]]), np.array([[0.0], [np.inf]])),
        (np.empty((0, 1)), np.empty((0, 1))),
    ],
)
def test_invalid_loss_shapes_and_values_are_rejected(y: np.ndarray, logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solution.binary_cross_entropy_from_logits(y, logits)


def test_guided_demo_exposes_broadcasting_trap() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "hidden: (3, 2)" in result.stdout
    assert "logits / probabilities: (3, 1) (3, 1)" in result.stdout
    assert "y vector / column: (3,) (3, 1)" in result.stdout
    assert "wrong error: (3, 3)" in result.stdout
    assert "correct error: (3, 1)" in result.stdout
