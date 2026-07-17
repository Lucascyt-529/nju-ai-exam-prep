import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "06_support_vector_machines" / "06_kernel_methods"
spec = importlib.util.spec_from_file_location("klda_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def xor_data():
    return (np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float),
            np.array([0, 0, 1, 1]))


def separable_data():
    return (np.array([[-2, 0], [-1, 0], [1, 0], [2, 0]], dtype=float),
            np.array([10, 10, 20, 20]))


def test_linear_polynomial_and_rbf_kernel_values_and_shapes():
    X = np.array([[1, 2], [0, 1]], dtype=float); Z = np.array([[2, 0]], dtype=float)
    np.testing.assert_allclose(solution.kernel_matrix(X, Z), [[2], [0]])
    np.testing.assert_allclose(solution.kernel_matrix(X, Z, kernel="polynomial", gamma=.5, degree=2, coef0=1), [[4], [1]])
    expected = np.exp(-.5 * np.array([[5.0], [5.0]]))
    np.testing.assert_allclose(solution.kernel_matrix(X, Z, kernel="rbf", gamma=.5), expected)


def test_rbf_training_kernel_is_symmetric_with_unit_diagonal():
    X, _ = xor_data(); K = solution.kernel_matrix(X, X, kernel="rbf", gamma=2)
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(np.diag(K), 1.0)


def test_representer_prediction_matches_matrix_vector_formula():
    K = np.array([[1.0, 2.0], [3.0, 4.0]]); coefficients = np.array([.5, -.25])
    np.testing.assert_allclose(solution.representer_prediction(K, coefficients, bias=.1), K @ coefficients + .1)


def test_kernel_scatter_shapes_symmetry_and_positive_semidefiniteness():
    X, y = separable_data(); K = solution.kernel_matrix(X, X)
    between, within = solution.kernel_scatter_matrices(K, y)
    assert between.shape == within.shape == (4, 4)
    np.testing.assert_allclose(between, between.T)
    np.testing.assert_allclose(within, within.T)
    assert np.linalg.eigvalsh(between).min() >= -1e-10
    assert np.linalg.eigvalsh(within).min() >= -1e-10


def test_linear_klda_fits_simple_separable_data_and_preserves_labels():
    X, y = separable_data(); model = solution.fit_kernel_lda(X, y, kernel="linear")
    np.testing.assert_array_equal(solution.predict(model, X), y)
    np.testing.assert_array_equal(model["classes"], [10, 20])
    assert model["coefficients"].shape == (4,)
    assert model["train_projections"].shape == (4,)


def test_rbf_klda_fits_xor_where_linear_kernel_has_no_mean_difference():
    X, y = xor_data()
    with pytest.raises(ValueError, match="有效判别方向"):
        solution.fit_kernel_lda(X, y, kernel="linear")
    model = solution.fit_kernel_lda(X, y, kernel="rbf", gamma=2.0, regularization=1e-4)
    np.testing.assert_array_equal(solution.predict(model, X), y)


def test_decision_scores_have_query_shape_and_match_prediction_sign():
    X, y = separable_data(); model = solution.fit_kernel_lda(X, y)
    query = np.array([[-3, 0], [3, 0]], dtype=float); scores = solution.decision_function(model, query)
    assert scores.shape == (2,) and scores[0] < 0 < scores[1]
    np.testing.assert_array_equal(solution.predict(model, query), [10, 20])


def test_rbf_predictions_are_translation_invariant_when_train_and_query_shift_together():
    X, y = xor_data(); shifted = X + np.array([10.0, -7.0])
    first = solution.fit_kernel_lda(X, y, kernel="rbf", gamma=2, regularization=1e-4)
    second = solution.fit_kernel_lda(shifted, y, kernel="rbf", gamma=2, regularization=1e-4)
    np.testing.assert_array_equal(solution.predict(first, X), solution.predict(second, shifted))


def test_regularization_handles_singular_duplicate_training_samples():
    X = np.array([[-1, 0], [-1, 0], [1, 0], [1, 0]], dtype=float); y = np.array([0, 0, 1, 1])
    model = solution.fit_kernel_lda(X, y, regularization=1e-3)
    assert np.all(np.isfinite(model["coefficients"]))
    np.testing.assert_array_equal(solution.predict(model, X), y)


def test_fisher_ratio_is_nonnegative_finite_or_infinite():
    X, y = separable_data(); ratio = solution.fisher_ratio(solution.fit_kernel_lda(X, y))
    assert ratio >= 0 and (np.isfinite(ratio) or np.isinf(ratio))


def test_fit_copies_training_data_and_does_not_modify_inputs():
    X, y = separable_data(); X_copy = X.copy(); y_copy = y.copy()
    model = solution.fit_kernel_lda(X, y); X[0, 0] = 99; y[0] = 99
    assert model["X_train"][0, 0] == pytest.approx(-2)
    assert model["classes"][0] == 10
    assert X_copy[0, 0] == -2 and y_copy[0] == 10


@pytest.mark.parametrize("bad", [np.array([1.0, 2.0]), np.empty((0, 2)), np.array([[1.0, np.nan]])])
def test_bad_feature_matrices_are_rejected(bad):
    with pytest.raises(ValueError): solution.kernel_matrix(bad, np.ones((1, 2)))


def test_bad_kernel_options_and_feature_mismatch_are_rejected():
    X = np.ones((2, 2))
    with pytest.raises(ValueError): solution.kernel_matrix(X, X, kernel="unknown")
    with pytest.raises(ValueError): solution.kernel_matrix(X, X, kernel="rbf", gamma=0)
    with pytest.raises(ValueError): solution.kernel_matrix(X, X, kernel="polynomial", degree=0)
    with pytest.raises(ValueError): solution.kernel_matrix(X, np.ones((2, 3)))


def test_bad_labels_regularization_and_query_shape_are_rejected():
    X, y = separable_data()
    with pytest.raises(ValueError): solution.fit_kernel_lda(X, np.zeros(4))
    with pytest.raises(ValueError): solution.fit_kernel_lda(X, y.reshape(-1, 1))
    with pytest.raises(ValueError): solution.fit_kernel_lda(X, y, regularization=-1)
    model = solution.fit_kernel_lda(X, y)
    with pytest.raises(ValueError): solution.predict(model, np.ones((2, 3)))


def test_bad_representer_shapes_and_bias_are_rejected():
    with pytest.raises(ValueError): solution.representer_prediction(np.eye(2), np.ones(3))
    with pytest.raises(ValueError): solution.representer_prediction(np.eye(2), np.ones(2), bias=np.nan)


def test_guided_demo_runs_and_reports_xor_predictions():
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "query-train kernel shape: (4, 4)" in result.stdout
    assert "coefficient shape: (4,)" in result.stdout
    assert "predictions: [0, 0, 1, 1]" in result.stdout
    assert "fisher ratio finite or infinite: True" in result.stdout
