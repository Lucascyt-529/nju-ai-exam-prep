import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "04_bp_training"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("network_training_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def xor_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return X, y


def test_apply_gradients_matches_formula_and_is_not_in_place() -> None:
    parameters = solution.initialize_parameters(2, 3, seed=4)
    original = {key: value.copy() for key, value in parameters.items()}
    gradients = {key: np.full_like(value, 0.25) for key, value in parameters.items()}
    updated = solution.apply_gradients(parameters, gradients, 0.2)
    for key in parameters:
        np.testing.assert_allclose(updated[key], original[key] - 0.05)
        np.testing.assert_array_equal(parameters[key], original[key])
        assert not np.shares_memory(updated[key], parameters[key])


def test_zero_epochs_returns_initial_parameters_and_one_loss() -> None:
    X, y = xor_data()
    trained, history = solution.train_network(
        X, y, n_hidden=3, learning_rate=0.5, epochs=0, seed=9
    )
    initial = solution.initialize_parameters(2, 3, seed=9)
    assert len(history) == 1 and np.isfinite(history[0])
    for key in trained:
        np.testing.assert_array_equal(trained[key], initial[key])


def test_linear_data_loss_decreases_and_predictions_are_correct() -> None:
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([[0.0], [0.0], [1.0], [1.0]])
    parameters, history = solution.train_network(
        X, y, n_hidden=3, learning_rate=0.5, epochs=500, seed=2
    )
    assert len(history) == 501
    assert history[-1] < history[0] * 0.2
    np.testing.assert_array_equal(solution.predict_labels(X, parameters), y.astype(int))


def test_fixed_configuration_learns_xor() -> None:
    X, y = xor_data()
    parameters, history = solution.train_network(
        X, y, n_hidden=4, learning_rate=1.0, epochs=2000, seed=0
    )
    prediction = solution.predict_labels(X, parameters)
    assert history[-1] < 0.01
    assert history[-1] < history[0]
    np.testing.assert_array_equal(prediction, y.astype(int))


def test_training_is_fully_reproducible() -> None:
    X, y = xor_data()
    first_parameters, first_history = solution.train_network(
        X, y, n_hidden=4, learning_rate=1.0, epochs=100, seed=3
    )
    second_parameters, second_history = solution.train_network(
        X, y, n_hidden=4, learning_rate=1.0, epochs=100, seed=3
    )
    assert first_history == second_history
    for key in first_parameters:
        np.testing.assert_array_equal(first_parameters[key], second_parameters[key])


def test_training_does_not_modify_X_or_y() -> None:
    X, y = xor_data()
    original_X, original_y = X.copy(), y.copy()
    solution.train_network(X, y, n_hidden=4, learning_rate=1.0, epochs=10, seed=0)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


def test_probability_and_label_shapes_are_columns() -> None:
    X, _ = xor_data()
    parameters = solution.initialize_parameters(2, 2, seed=0)
    probabilities = solution.predict_probabilities(X, parameters)
    labels = solution.predict_labels(X, parameters)
    assert probabilities.shape == labels.shape == (4, 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(np.unique(labels)).issubset({0, 1})


def test_threshold_tie_is_classified_as_one() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    parameters = {
        "W1": np.zeros((2, 2)),
        "b1": np.zeros(2),
        "W2": np.zeros((2, 1)),
        "b2": np.zeros(1),
    }
    np.testing.assert_allclose(solution.predict_probabilities(X, parameters), 0.5)
    np.testing.assert_array_equal(solution.predict_labels(X, parameters), np.ones((2, 1)))


@pytest.mark.parametrize(
    "y",
    [
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([[0.0], [1.0], [1.0]]),
        np.array([[0.0], [2.0], [1.0], [0.0]]),
        np.array([[0.0], [1.0], [np.nan], [0.0]]),
    ],
)
def test_invalid_training_labels_are_rejected(y: np.ndarray) -> None:
    X, _ = xor_data()
    with pytest.raises(ValueError):
        solution.train_network(X, y, n_hidden=4, learning_rate=1.0, epochs=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_hidden": 0, "learning_rate": 1.0, "epochs": 1},
        {"n_hidden": True, "learning_rate": 1.0, "epochs": 1},
        {"n_hidden": 4, "learning_rate": 0.0, "epochs": 1},
        {"n_hidden": 4, "learning_rate": np.nan, "epochs": 1},
        {"n_hidden": 4, "learning_rate": 1.0, "epochs": -1},
        {"n_hidden": 4, "learning_rate": 1.0, "epochs": 1.5},
        {"n_hidden": 4, "learning_rate": 1.0, "epochs": 1, "seed": True},
    ],
)
def test_invalid_training_hyperparameters_are_rejected(kwargs) -> None:
    X, y = xor_data()
    with pytest.raises(ValueError):
        solution.train_network(X, y, **kwargs)


def test_invalid_gradient_dictionary_is_rejected() -> None:
    parameters = solution.initialize_parameters(2, 2, seed=0)
    missing = {key: np.zeros_like(value) for key, value in parameters.items() if key != "b2"}
    with pytest.raises(ValueError):
        solution.apply_gradients(parameters, missing, 0.1)

    wrong_shape = {key: np.zeros_like(value) for key, value in parameters.items()}
    wrong_shape["W2"] = np.zeros(2)
    with pytest.raises(ValueError):
        solution.apply_gradients(parameters, wrong_shape, 0.1)

    nonfinite = {key: np.zeros_like(value) for key, value in parameters.items()}
    nonfinite["b2"][0] = np.nan
    with pytest.raises(ValueError):
        solution.apply_gradients(parameters, nonfinite, 0.1)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, True])
def test_invalid_prediction_thresholds_are_rejected(threshold) -> None:
    X, _ = xor_data()
    parameters = solution.initialize_parameters(2, 2, seed=0)
    with pytest.raises(ValueError):
        solution.predict_labels(X, parameters, threshold=threshold)


def test_guided_demo_trains_xor_end_to_end() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "standard BP sample orders:" in result.stdout
    assert "standard BP losses:" in result.stdout
    assert "accumulated BP losses:" in result.stdout
    assert "same W1 after two epochs: False" in result.stdout
    assert "X / y: (4, 2) (4, 1)" in result.stdout
    assert "history length: 2001" in result.stdout
    assert "prediction: [0 1 1 0]" in result.stdout
