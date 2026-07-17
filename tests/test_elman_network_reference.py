import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "09_elman_network"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("elman_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def memory_parameters() -> dict[str, np.ndarray]:
    return {
        "Wx": np.array([[1.0]]),
        "Wh": np.array([[0.8]]),
        "bh": np.zeros(1),
        "Wy": np.array([[1.0]]),
        "by": np.zeros(1),
    }


def test_initialization_shapes_and_reproducibility() -> None:
    first = solution.initialize_elman_parameters(2, 3, 4, seed=7)
    second = solution.initialize_elman_parameters(2, 3, 4, seed=7)
    assert first["Wx"].shape == (2, 3)
    assert first["Wh"].shape == (3, 3)
    assert first["bh"].shape == (3,)
    assert first["Wy"].shape == (3, 4)
    assert first["by"].shape == (4,)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_single_step_matches_manual_recurrence() -> None:
    parameters = memory_parameters()
    state, output = solution.elman_step(
        np.array([1.0]), np.array([0.5]), parameters
    )
    expected_state = np.tanh(1.0 + 0.8 * 0.5)
    np.testing.assert_allclose(state, [expected_state])
    np.testing.assert_allclose(output, [expected_state])


def test_sequence_shapes_include_initial_state() -> None:
    X = np.array([[1.0], [0.0], [-1.0]])
    result = solution.forward_sequence(X, memory_parameters())
    assert result["states"].shape == (4, 1)
    assert result["outputs"].shape == (3, 1)
    np.testing.assert_array_equal(result["states"][0], [0.0])


def test_different_histories_change_same_current_input_output() -> None:
    parameters = memory_parameters()
    positive = solution.forward_sequence(np.array([[1.0], [0.0]]), parameters)
    negative = solution.forward_sequence(np.array([[-1.0], [0.0]]), parameters)
    assert positive["outputs"][-1, 0] > 0
    assert negative["outputs"][-1, 0] < 0
    assert positive["outputs"][-1, 0] != pytest.approx(negative["outputs"][-1, 0])


def test_segmented_continuation_equals_whole_sequence() -> None:
    X = np.array([[1.0], [0.5], [-0.5], [0.0]])
    parameters = memory_parameters()
    whole = solution.forward_sequence(X, parameters)
    first = solution.forward_sequence(X[:2], parameters)
    second = solution.forward_sequence(
        X[2:], parameters, initial_state=first["states"][-1]
    )
    combined_outputs = np.vstack([first["outputs"], second["outputs"]])
    np.testing.assert_allclose(combined_outputs, whole["outputs"])
    np.testing.assert_allclose(second["states"][-1], whole["states"][-1])


def test_resetting_state_changes_second_segment() -> None:
    X = np.array([[1.0], [0.5], [0.0]])
    parameters = memory_parameters()
    first = solution.forward_sequence(X[:2], parameters)
    continued = solution.forward_sequence(
        X[2:], parameters, initial_state=first["states"][-1]
    )
    reset = solution.forward_sequence(X[2:], parameters)
    assert continued["outputs"][0, 0] != pytest.approx(reset["outputs"][0, 0])


def test_zero_recurrent_weights_remove_history_dependence() -> None:
    parameters = memory_parameters()
    parameters["Wh"] = np.zeros((1, 1))
    positive = solution.forward_sequence(np.array([[1.0], [0.0]]), parameters)
    negative = solution.forward_sequence(np.array([[-1.0], [0.0]]), parameters)
    np.testing.assert_allclose(positive["outputs"][-1], negative["outputs"][-1])


def test_sequence_mse_matches_manual_value_and_rejects_broadcasting() -> None:
    targets = np.array([[0.0], [1.0]])
    outputs = np.array([[1.0], [1.0]])
    assert solution.sequence_mean_squared_error(targets, outputs) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        solution.sequence_mean_squared_error(targets.ravel(), outputs)


def test_forward_does_not_modify_inputs_or_initial_state() -> None:
    X = np.array([[1.0], [0.0]])
    initial = np.array([0.25])
    original_X, original_initial = X.copy(), initial.copy()
    solution.forward_sequence(X, memory_parameters(), initial_state=initial)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(initial, original_initial)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.initialize_elman_parameters(0, 2, 1),
        lambda: solution.initialize_elman_parameters(1, True, 1),
        lambda: solution.initialize_elman_parameters(1, 2, 1, seed=True),
        lambda: solution.forward_sequence(np.array([1.0, 2.0]), memory_parameters()),
        lambda: solution.forward_sequence(np.empty((0, 1)), memory_parameters()),
        lambda: solution.forward_sequence(
            np.ones((2, 1)), memory_parameters(), initial_state=np.zeros(2)
        ),
        lambda: solution.elman_step(
            np.array([[1.0]]), np.zeros(1), memory_parameters()
        ),
        lambda: solution.elman_step(
            np.array([1.0]), np.zeros(2), memory_parameters()
        ),
    ],
)
def test_invalid_elman_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_wrong_parameter_shapes_are_rejected() -> None:
    parameters = memory_parameters()
    parameters["Wh"] = np.eye(2)
    with pytest.raises(ValueError):
        solution.forward_sequence(np.ones((2, 1)), parameters)


def test_guided_demo_exposes_context_memory() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "X: (2, 1)" in result.stdout
    assert "states / outputs: (3, 1) (2, 1)" in result.stdout
    assert "same final input: [0.] [0.]" in result.stdout
    assert "positive-history final output: [" in result.stdout
    assert "negative-history final output: [-" in result.stdout
