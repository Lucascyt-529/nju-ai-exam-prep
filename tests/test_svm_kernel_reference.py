import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "06_support_vector_machines" / "04_kernel_svm"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("svm_kernel_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def xor_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([-1, 1, 1, -1])
    return X, y


def test_linear_kernel_matches_matrix_product_and_shape() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    actual = solution.kernel_matrix(X, Z, kernel="linear")
    np.testing.assert_allclose(actual, X @ Z.T)
    assert actual.shape == (2, 3)


def test_polynomial_kernel_matches_hand_calculation() -> None:
    X = np.array([[1.0, 2.0]])
    Z = np.array([[3.0, 4.0], [-1.0, 1.0]])
    actual = solution.kernel_matrix(
        X, Z, kernel="polynomial", degree=2, gamma=0.5, coef0=1.0
    )
    np.testing.assert_allclose(
        actual, [[(0.5 * 11 + 1) ** 2, (0.5 * 1 + 1) ** 2]]
    )


def test_rbf_kernel_matches_squared_distance_formula() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    Z = np.array([[0.0, 1.0]])
    actual = solution.kernel_matrix(X, Z, kernel="rbf", gamma=0.5)
    np.testing.assert_allclose(actual[:, 0], [np.exp(-0.5), np.exp(-1.0)])


@pytest.mark.parametrize("kernel", ["linear", "polynomial", "rbf"])
def test_training_gram_is_symmetric_positive_semidefinite(kernel: str) -> None:
    X = np.array([[-1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    gram = solution.kernel_matrix(X, X, kernel=kernel, gamma=0.7, degree=2)
    np.testing.assert_allclose(gram, gram.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(gram)) >= -1e-10


def test_rbf_diagonal_is_one_and_entries_are_bounded() -> None:
    X = np.array([[-2.0], [0.0], [3.0]])
    gram = solution.kernel_matrix(X, X, kernel="rbf", gamma=2.0)
    np.testing.assert_allclose(np.diag(gram), 1.0)
    assert np.all((gram > 0.0) & (gram <= 1.0))


def test_default_gamma_is_inverse_feature_count() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    default = solution.kernel_matrix(X, X, kernel="rbf")
    explicit = solution.kernel_matrix(X, X, kernel="rbf", gamma=0.5)
    np.testing.assert_allclose(default, explicit)


def test_rbf_svm_fits_xor_while_linear_kernel_does_not() -> None:
    X, y = xor_data()
    linear = solution.fit_kernel_svm_smo(X, y, C=10.0, kernel="linear", tolerance=1e-6, max_passes=30)
    rbf = solution.fit_kernel_svm_smo(X, y, C=10.0, kernel="rbf", gamma=2.0, tolerance=1e-6, max_passes=30)
    assert np.mean(solution.predict_labels(linear, X) == y) < 1.0
    np.testing.assert_array_equal(solution.predict_labels(rbf, X), y)


def test_polynomial_degree_two_also_represents_xor() -> None:
    X, y = xor_data()
    model = solution.fit_kernel_svm_smo(
        X, y, C=10.0, kernel="polynomial", degree=2, gamma=1.0,
        coef0=1.0, tolerance=1e-6, max_passes=30,
    )
    np.testing.assert_array_equal(solution.predict_labels(model, X), y)


def test_support_only_prediction_matches_full_expansion() -> None:
    X, y = xor_data()
    model = solution.fit_kernel_svm_smo(X, y, C=10.0, kernel="rbf", gamma=2.0, tolerance=1e-6, max_passes=30)
    query = np.array([[0.25, 0.25], [0.25, 0.75], [2.0, 2.0]])
    np.testing.assert_allclose(
        solution.decision_function(model, query, support_only=True),
        solution.decision_function(model, query, support_only=False),
    )


def test_alpha_constraints_kkt_and_dual_history_hold() -> None:
    X, y = xor_data()
    model = solution.fit_kernel_svm_smo(X, y, C=10.0, kernel="rbf", gamma=2.0, tolerance=1e-6, max_passes=30)
    assert np.all((model["alphas"] >= -1e-10) & (model["alphas"] <= 10.0 + 1e-10))
    assert float(model["alphas"] @ y) == pytest.approx(0.0, abs=1e-10)
    assert np.max(solution.kkt_residuals(model)) < 1e-5
    assert np.all(np.diff(model["dual_history"]) >= -1e-10)


def test_training_is_deterministic_and_does_not_modify_inputs() -> None:
    X, y = xor_data()
    X0, y0 = X.copy(), y.copy()
    first = solution.fit_kernel_svm_smo(X, y, C=10.0, kernel="rbf", gamma=2.0)
    second = solution.fit_kernel_svm_smo(X, y, C=10.0, kernel="rbf", gamma=2.0)
    np.testing.assert_array_equal(first["alphas"], second["alphas"])
    assert first["bias"] == second["bias"]
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel": "unknown"},
        {"kernel": "rbf", "gamma": 0.0},
        {"kernel": "polynomial", "degree": 0},
        {"kernel": "polynomial", "coef0": np.inf},
    ],
)
def test_kernel_matrix_rejects_invalid_configuration(kwargs) -> None:
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        solution.kernel_matrix(X, X, **kwargs)


def test_kernel_matrix_rejects_bad_shape_dimension_and_values() -> None:
    with pytest.raises(ValueError):
        solution.kernel_matrix(np.array([1.0, 2.0]), np.array([[1.0]]))
    with pytest.raises(ValueError):
        solution.kernel_matrix(np.ones((2, 2)), np.ones((3, 1)))
    with pytest.raises(ValueError):
        solution.kernel_matrix(np.array([[np.nan]]), np.array([[1.0]]))


@pytest.mark.parametrize("C", [0.0, -1.0, np.inf, True])
def test_fit_rejects_invalid_C(C) -> None:
    X, y = xor_data()
    with pytest.raises(ValueError):
        solution.fit_kernel_svm_smo(X, y, C=C)


def test_fit_rejects_bad_labels_and_query_feature_count() -> None:
    X, y = xor_data()
    with pytest.raises(ValueError):
        solution.fit_kernel_svm_smo(X, np.array([0, 1, 1, 0]))
    model = solution.fit_kernel_svm_smo(X, y, C=10.0, gamma=2.0)
    with pytest.raises(ValueError):
        solution.decision_function(model, np.ones((2, 3)))


def test_guided_demo_runs_and_shows_xor_predictions() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "kernel=linear" in result.stdout
    assert "kernel=rbf" in result.stdout
    assert "predictions: [-1, 1, 1, -1]" in result.stdout
