import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "06_support_vector_machines" / "05_epsilon_svr"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("epsilon_svr_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def linear_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(-2.0, 3.0).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0 + np.array([0.0, 0.1, -0.05, 0.05, 0.0])
    return X, y


def test_epsilon_loss_matches_hand_calculation() -> None:
    y = np.array([0.0, 0.0, 0.0, 0.0])
    prediction = np.array([0.05, -0.1, 0.3, -0.5])
    np.testing.assert_allclose(
        solution.epsilon_insensitive_loss(y, prediction, 0.1),
        [0.0, 0.0, 0.2, 0.4],
    )


def test_tube_regions_cover_inside_boundary_and_outside() -> None:
    residuals = np.array([0.0, 0.1, -0.1, 0.2])
    assert solution.tube_regions(residuals, 0.1).tolist() == [
        "inside_tube", "on_tube", "on_tube", "outside_tube"
    ]


def test_dual_objective_matches_hand_calculation() -> None:
    beta = np.array([0.5, -0.5])
    y = np.array([1.0, -1.0])
    gram = np.eye(2)
    expected = -0.5 * 0.5 - 0.1 * 1.0 + 1.0
    assert solution.dual_objective(beta, y, gram, 0.1) == pytest.approx(expected)


def test_linear_svr_keeps_training_residuals_inside_tube() -> None:
    X, y = linear_data()
    model = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.1, tolerance=1e-9)
    residuals = y - solution.decision_function(model, X)
    assert np.max(np.abs(residuals)) <= 0.1 + 1e-7
    assert solution.decision_function(model, X).shape == (5,)


def test_beta_box_sum_constraint_and_support_vectors() -> None:
    X, y = linear_data()
    model = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.1, tolerance=1e-9)
    assert np.all(np.abs(model["beta"]) <= 10.0 + 1e-10)
    assert float(np.sum(model["beta"])) == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_array_equal(solution.support_vector_indices(model), [0, 3])


def test_linear_solution_kkt_and_dual_history() -> None:
    X, y = linear_data()
    model = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.1, tolerance=1e-9)
    assert np.max(solution.kkt_residuals(model)) < 1e-7
    assert np.all(np.diff(model["dual_history"]) >= -1e-12)
    assert model["updates"] > 0


def test_support_only_prediction_matches_full_expansion() -> None:
    X, y = linear_data()
    model = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.1, tolerance=1e-9)
    query = np.array([[-3.0], [0.5], [4.0]])
    np.testing.assert_allclose(
        solution.decision_function(model, query, support_only=True),
        solution.decision_function(model, query, support_only=False),
    )


def test_rbf_svr_fits_curved_targets_within_epsilon() -> None:
    X = np.linspace(-1.0, 1.0, 7).reshape(-1, 1)
    y = X[:, 0] ** 2
    model = solution.fit_epsilon_svr(
        X, y, C=20.0, epsilon=0.03, kernel="rbf", gamma=3.0,
        tolerance=1e-12, max_passes=200,
    )
    prediction = solution.decision_function(model, X)
    assert np.max(np.abs(y - prediction)) <= 0.03001
    assert np.max(solution.kkt_residuals(model)) < 2e-6


def test_constant_target_uses_bias_without_support_vectors() -> None:
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([2.0, 2.0, 2.0])
    model = solution.fit_epsilon_svr(X, y, epsilon=0.1)
    assert solution.support_vector_indices(model).size == 0
    np.testing.assert_allclose(solution.decision_function(model, X), 2.0)


def test_larger_epsilon_can_reduce_support_count() -> None:
    X, y = linear_data()
    narrow = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.01, tolerance=1e-9)
    wide = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.2, tolerance=1e-9)
    assert solution.support_vector_indices(wide).size <= solution.support_vector_indices(narrow).size


def test_training_is_deterministic_and_does_not_modify_inputs() -> None:
    X, y = linear_data()
    X0, y0 = X.copy(), y.copy()
    first = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.1)
    second = solution.fit_epsilon_svr(X, y, C=10.0, epsilon=0.1)
    np.testing.assert_array_equal(first["beta"], second["beta"])
    assert first["bias"] == second["bias"]
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)


@pytest.mark.parametrize(
    ("y", "prediction"),
    [
        (np.array([[1.0]]), np.array([1.0])),
        (np.array([1.0]), np.array([[1.0]])),
        (np.array([1.0]), np.array([np.nan])),
        (np.array([1.0, 2.0]), np.array([1.0])),
    ],
)
def test_loss_rejects_bad_shapes_values_and_mismatch(y, prediction) -> None:
    with pytest.raises(ValueError):
        solution.epsilon_insensitive_loss(y, prediction, 0.1)


@pytest.mark.parametrize("epsilon", [-0.1, np.inf, True])
def test_loss_rejects_invalid_epsilon(epsilon) -> None:
    with pytest.raises(ValueError):
        solution.epsilon_insensitive_loss(np.array([1.0]), np.array([1.0]), epsilon)


def test_fit_rejects_column_target_invalid_kernel_and_query_shape() -> None:
    X, y = linear_data()
    with pytest.raises(ValueError):
        solution.fit_epsilon_svr(X, y[:, None])
    with pytest.raises(ValueError):
        solution.fit_epsilon_svr(X, y, kernel="polynomial")
    model = solution.fit_epsilon_svr(X, y)
    with pytest.raises(ValueError):
        solution.decision_function(model, np.ones((2, 2)))


@pytest.mark.parametrize("C", [0.0, -1.0, np.inf, True])
def test_fit_rejects_invalid_C(C) -> None:
    X, y = linear_data()
    with pytest.raises(ValueError):
        solution.fit_epsilon_svr(X, y, C=C)


def test_guided_demo_runs_and_reports_tube_and_supports() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "prediction shape: (5,)" in result.stdout
    assert "beta sum: 0.0" in result.stdout
    assert "tube regions:" in result.stdout
    assert "support indices: [0, 3]" in result.stdout
