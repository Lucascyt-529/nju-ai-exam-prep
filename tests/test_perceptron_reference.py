import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "01_perceptron"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("perceptron_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def separable_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([-1, -1, 1, 1])
    return X, y


def test_decision_function_is_matrix_vector_product() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([2.0, -1.0])
    scores = solution.decision_function(X, weights, 0.5)
    np.testing.assert_allclose(scores, [0.5, 2.5])
    assert scores.shape == (2,)


def test_prediction_uses_positive_label_on_zero_score() -> None:
    X = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    prediction = solution.predict_perceptron(X, np.array([1.0, 0.0]), 0.0)
    np.testing.assert_array_equal(prediction, [1, -1, 1])
    assert prediction.shape == (3,)


def test_first_epoch_matches_hand_calculation() -> None:
    X = np.array([[-2.0, -1.0], [2.0, 1.0]])
    y = np.array([-1, 1])
    weights, bias, history = solution.train_perceptron(
        X, y, learning_rate=0.5, max_epochs=1
    )
    np.testing.assert_allclose(weights, [1.0, 0.5])
    assert bias == pytest.approx(-0.5)
    assert history == [1]


def test_linearly_separable_data_converges() -> None:
    X, y = separable_data()
    weights, bias, history = solution.train_perceptron(X, y, max_epochs=50)
    np.testing.assert_array_equal(solution.predict_perceptron(X, weights, bias), y)
    assert history[-1] == 0
    assert len(history) < 50


def test_bias_learns_boundary_away_from_origin() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([-1, -1, 1, 1])
    weights, bias, history = solution.train_perceptron(X, y, max_epochs=100)
    np.testing.assert_array_equal(solution.predict_perceptron(X, weights, bias), y)
    assert bias != 0.0
    assert history[-1] == 0


def test_xor_reaches_epoch_limit_without_false_convergence() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([-1, 1, 1, -1])
    _, _, history = solution.train_perceptron(X, y, max_epochs=12)
    assert len(history) == 12
    assert all(updates > 0 for updates in history)


def test_training_is_deterministic() -> None:
    X, y = separable_data()
    first = solution.train_perceptron(X, y, learning_rate=0.25, max_epochs=20)
    second = solution.train_perceptron(X, y, learning_rate=0.25, max_epochs=20)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_inputs_are_not_modified() -> None:
    X, y = separable_data()
    original_X = X.copy()
    original_y = y.copy()
    solution.train_perceptron(X, y)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "X",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.array([[1.0, np.nan], [2.0, 3.0]]),
        np.array([["a", "b"]]),
    ],
)
def test_invalid_feature_arrays_are_rejected(X: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solution.decision_function(X, np.ones(2), 0.0)


@pytest.mark.parametrize(
    "y",
    [
        np.array([[-1], [1], [1], [-1]]),
        np.array([-1, 1, 1]),
        np.array([0, 0, 1, 1]),
        np.array([-1, -1, -1, -1]),
        np.array([-1.0, 1.0, np.nan, 1.0]),
    ],
)
def test_invalid_labels_are_rejected(y: np.ndarray) -> None:
    X = np.eye(4, 2)
    with pytest.raises(ValueError):
        solution.train_perceptron(X, y)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.decision_function(np.eye(2), np.ones((2, 1)), 0.0),
        lambda: solution.decision_function(np.eye(2), np.ones(2), np.inf),
        lambda: solution.train_perceptron(*separable_data(), learning_rate=0.0),
        lambda: solution.train_perceptron(*separable_data(), learning_rate=np.nan),
        lambda: solution.train_perceptron(*separable_data(), max_epochs=0),
        lambda: solution.train_perceptron(*separable_data(), max_epochs=True),
    ],
)
def test_invalid_parameters_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_guided_demo_reports_shapes_and_first_update() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "X shape: (2, 2)" in result.stdout
    assert "y shape: (2,)" in result.stdout
    assert "weights shape: (2,)" in result.stdout
    assert "updated weights: [1.  0.5]" in result.stdout
    assert "updated bias: -0.5" in result.stdout
