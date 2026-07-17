import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "06_support_vector_machines" / "02_linear_smo"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("svm_smo_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def separable_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1, -1, 1, 1])
    return X, y


def test_linear_gram_matrix_matches_dot_products() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    gram = solution.linear_kernel_matrix(X, Z)
    np.testing.assert_allclose(gram, X @ Z.T)
    assert gram.shape == (2, 3)


def test_dual_objective_matches_hand_calculation() -> None:
    alphas = np.array([0.5, 0.5])
    y = np.array([-1.0, 1.0])
    gram = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert solution.dual_objective(alphas, y, gram) == pytest.approx(0.5)


def test_smo_fits_separable_training_data() -> None:
    X, y = separable_data()
    model = solution.fit_linear_svm_smo(X, y, C=10.0)
    np.testing.assert_array_equal(solution.predict_labels(model, X), y)
    np.testing.assert_allclose(solution.linear_weights(model), [1.0], atol=1e-6)
    assert model["bias"] == pytest.approx(0.0, abs=1e-6)


def test_alpha_box_and_equality_constraints_hold() -> None:
    X, y = separable_data()
    model = solution.fit_linear_svm_smo(X, y, C=10.0)
    assert np.all(model["alphas"] >= -1e-10)
    assert np.all(model["alphas"] <= 10.0 + 1e-10)
    assert float(model["alphas"] @ y) == pytest.approx(0.0, abs=1e-10)


def test_support_vectors_are_the_two_closest_points() -> None:
    X, y = separable_data()
    model = solution.fit_linear_svm_smo(X, y, C=10.0)
    np.testing.assert_array_equal(solution.support_vector_indices(model), [1, 2])
    np.testing.assert_allclose(model["alphas"][[1, 2]], [0.5, 0.5], atol=1e-6)


def test_kkt_residuals_are_small() -> None:
    X, y = separable_data()
    model = solution.fit_linear_svm_smo(X, y, C=10.0, tolerance=1e-5)
    assert np.max(solution.kkt_residuals(model)) < 1e-5


def test_primal_and_dual_objectives_agree_on_separable_solution() -> None:
    X, y = separable_data()
    model = solution.fit_linear_svm_smo(X, y, C=10.0, tolerance=1e-5)
    gram = solution.linear_kernel_matrix(X, X)
    dual = solution.dual_objective(model["alphas"], y, gram)
    assert solution.primal_objective(model) == pytest.approx(dual, abs=1e-6)


def test_dual_history_is_nondecreasing() -> None:
    X, y = separable_data()
    model = solution.fit_linear_svm_smo(X, y, C=10.0)
    assert model["dual_history"].shape == (model["iterations"] + 1,)
    assert np.all(np.diff(model["dual_history"]) >= -1e-10)
    assert model["updates"] > 0


def test_duplicate_opposite_labels_use_degenerate_endpoint_update() -> None:
    X = np.array([[0.0], [0.0]])
    y = np.array([-1, 1])
    model = solution.fit_linear_svm_smo(X, y, C=1.0)
    np.testing.assert_allclose(model["alphas"], [1.0, 1.0])
    assert float(model["alphas"] @ y) == pytest.approx(0.0)
    assert solution.dual_objective(
        model["alphas"], y, solution.linear_kernel_matrix(X, X)
    ) == pytest.approx(2.0)


def test_fit_is_deterministic() -> None:
    X, y = separable_data()
    first = solution.fit_linear_svm_smo(X, y, C=10.0)
    second = solution.fit_linear_svm_smo(X, y, C=10.0)
    np.testing.assert_array_equal(first["alphas"], second["alphas"])
    assert first["bias"] == second["bias"]
    np.testing.assert_array_equal(first["dual_history"], second["dual_history"])


def test_training_inputs_are_not_modified() -> None:
    X, y = separable_data()
    original_X, original_y = X.copy(), y.copy()
    solution.fit_linear_svm_smo(X, y, C=10.0)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.linear_kernel_matrix(np.array([1.0]), np.ones((1, 1))),
        lambda: solution.linear_kernel_matrix(np.ones((1, 2)), np.ones((1, 3))),
        lambda: solution.dual_objective(
            np.array([-1.0, 1.0]), np.array([-1, 1]), np.eye(2)
        ),
        lambda: solution.fit_linear_svm_smo(
            np.ones((2, 1)), np.array([0, 1])
        ),
        lambda: solution.fit_linear_svm_smo(*separable_data(), C=0.0),
        lambda: solution.fit_linear_svm_smo(*separable_data(), tolerance=0.0),
        lambda: solution.fit_linear_svm_smo(*separable_data(), max_passes=0),
        lambda: solution.fit_linear_svm_smo(*separable_data(), max_iterations=True),
    ],
)
def test_invalid_smo_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_guided_demo_reports_constraints_and_objectives() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "sum alpha*y: 0.0" in result.stdout
    assert "weights / bias: [1.] 0.0" in result.stdout
    assert "support indices: [1 2]" in result.stdout
    assert "prediction: [-1 -1  1  1]" in result.stdout
    assert "primal / dual: 0.5 0.5" in result.stdout
