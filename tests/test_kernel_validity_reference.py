import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "06_support_vector_machines" / "07_kernel_validity"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("kernel_validity_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def sample_data() -> np.ndarray:
    return np.array([[-2.0, 0.5], [0.0, 1.0], [1.5, -1.0], [3.0, 2.0]])


def test_linear_kernel_shape_and_values() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Z = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    actual = solution.kernel_matrix(X, Z)
    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, X @ Z.T)


def test_polynomial_kernel_matches_hand_calculation() -> None:
    X = np.array([[1.0, 2.0]])
    Z = np.array([[3.0, 4.0], [-1.0, 1.0]])
    actual = solution.kernel_matrix(
        X, Z, kernel="polynomial", gamma=0.5, degree=2, coef0=1.0
    )
    np.testing.assert_allclose(actual, [[42.25, 2.25]])


def test_rbf_and_laplacian_match_distance_formula() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Z = np.array([[0.0, 0.0]])
    rbf = solution.kernel_matrix(X, Z, kernel="rbf", gamma=0.2)
    laplacian = solution.kernel_matrix(X, Z, kernel="laplacian", gamma=0.2)
    np.testing.assert_allclose(rbf[:, 0], [1.0, np.exp(-5.0)])
    np.testing.assert_allclose(laplacian[:, 0], [1.0, np.exp(-1.0)])


def test_sigmoid_matches_definition() -> None:
    X = np.array([[1.0], [2.0]])
    actual = solution.kernel_matrix(
        X, X, kernel="sigmoid", gamma=0.5, coef0=-1.0
    )
    np.testing.assert_allclose(actual, np.tanh(0.5 * (X @ X.T) - 1.0))


@pytest.mark.parametrize("kind", ["linear", "polynomial", "rbf", "laplacian"])
def test_common_valid_kernels_have_psd_training_gram(kind: str) -> None:
    X = sample_data()
    gram = solution.kernel_matrix(
        X, X, kernel=kind, gamma=0.7, degree=2, coef0=1.0
    )
    report = solution.gram_diagnostics(gram)
    assert report["symmetric"] is True
    assert report["positive_semidefinite"] is True
    assert report["minimum_eigenvalue"] >= -1e-10


def test_sigmoid_candidate_provides_finite_counterexample() -> None:
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    gram = solution.kernel_matrix(
        X, X, kernel="sigmoid", gamma=1.0, coef0=-1.0
    )
    report = solution.gram_diagnostics(gram)
    assert report["symmetric"] is True
    assert report["positive_semidefinite"] is False
    assert report["minimum_eigenvalue"] < -0.2


def test_diagnostics_rejects_asymmetric_matrix() -> None:
    report = solution.gram_diagnostics(np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert report["symmetric"] is False
    assert report["positive_semidefinite"] is False
    assert report["symmetry_error"] == pytest.approx(2.0)


def test_quadratic_form_exhibits_negative_direction() -> None:
    gram = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert solution.quadratic_form(gram, np.array([1.0, -1.0])) == pytest.approx(-2.0)


def test_quadratic_form_nonnegative_for_psd_examples() -> None:
    X = sample_data()
    gram = solution.kernel_matrix(X, X, kernel="rbf", gamma=0.5)
    for coefficients in [np.ones(4), np.array([1.0, -2.0, 0.5, 3.0])]:
        assert solution.quadratic_form(gram, coefficients) >= -1e-10


def test_finite_feature_coordinates_reconstructs_psd_gram() -> None:
    X = sample_data()
    gram = solution.kernel_matrix(X, X, kernel="rbf", gamma=0.5)
    coordinates = solution.finite_feature_coordinates(gram)
    assert coordinates.shape == (4, 4)
    np.testing.assert_allclose(coordinates @ coordinates.T, gram, atol=1e-10)


def test_finite_feature_coordinates_drops_zero_eigenvalues() -> None:
    gram = np.ones((3, 3))
    coordinates = solution.finite_feature_coordinates(gram)
    assert coordinates.shape == (3, 1)
    np.testing.assert_allclose(coordinates @ coordinates.T, gram, atol=1e-10)


def test_finite_feature_coordinates_handles_zero_gram() -> None:
    coordinates = solution.finite_feature_coordinates(np.zeros((3, 3)))
    assert coordinates.shape == (3, 0)
    np.testing.assert_allclose(coordinates @ coordinates.T, np.zeros((3, 3)))


def test_finite_feature_coordinates_rejects_indefinite_gram() -> None:
    with pytest.raises(ValueError, match="半正定"):
        solution.finite_feature_coordinates(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_positive_weighted_sum_matches_formula_and_is_psd() -> None:
    X = sample_data()
    linear = solution.kernel_matrix(X, X)
    rbf = solution.kernel_matrix(X, X, kernel="rbf", gamma=0.5)
    actual = solution.positive_weighted_sum(
        [linear, rbf], np.array([0.25, 0.75])
    )
    np.testing.assert_allclose(actual, 0.25 * linear + 0.75 * rbf)
    assert solution.gram_diagnostics(actual)["positive_semidefinite"] is True


def test_product_kernel_grams_matches_hadamard_product_and_is_psd() -> None:
    X = sample_data()
    rbf = solution.kernel_matrix(X, X, kernel="rbf", gamma=0.5)
    laplacian = solution.kernel_matrix(X, X, kernel="laplacian", gamma=0.8)
    actual = solution.product_kernel_grams([rbf, laplacian])
    np.testing.assert_allclose(actual, rbf * laplacian)
    assert solution.gram_diagnostics(actual)["positive_semidefinite"] is True


def test_scale_kernel_gram_matches_congruence_and_is_psd() -> None:
    X = sample_data()
    gram = solution.kernel_matrix(X, X, kernel="rbf", gamma=0.5)
    values = np.array([-2.0, 0.0, 0.5, 3.0])
    actual = solution.scale_kernel_gram(gram, values)
    expected = np.diag(values) @ gram @ np.diag(values)
    np.testing.assert_allclose(actual, expected)
    assert solution.gram_diagnostics(actual)["positive_semidefinite"] is True


def test_negative_kernel_weight_can_destroy_psd() -> None:
    first = np.eye(2)
    second = 2.0 * np.eye(2)
    candidate = first - second
    assert solution.gram_diagnostics(candidate)["positive_semidefinite"] is False
    with pytest.raises(ValueError, match="正有限"):
        solution.positive_weighted_sum(
            [first, second], np.array([1.0, -1.0])
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel": "unknown"},
        {"kernel": "rbf", "gamma": 0.0},
        {"kernel": "polynomial", "degree": 0},
        {"kernel": "polynomial", "coef0": np.inf},
    ],
)
def test_kernel_matrix_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        solution.kernel_matrix(X, X, **kwargs)


def test_kernel_matrix_rejects_bad_shapes_dimensions_and_values() -> None:
    with pytest.raises(ValueError):
        solution.kernel_matrix(np.array([1.0]), np.array([[1.0]]))
    with pytest.raises(ValueError):
        solution.kernel_matrix(np.ones((2, 2)), np.ones((3, 1)))
    with pytest.raises(ValueError):
        solution.kernel_matrix(np.array([[np.nan]]), np.array([[1.0]]))


def test_diagnostics_rejects_non_square_and_bad_tolerances() -> None:
    with pytest.raises(ValueError, match="方阵"):
        solution.gram_diagnostics(np.ones((2, 3)))
    with pytest.raises(ValueError):
        solution.gram_diagnostics(np.eye(2), eigen_tolerance=0.0)


def test_quadratic_form_rejects_wrong_coefficient_shape() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        solution.quadratic_form(np.eye(2), np.ones((2, 1)))


def test_combination_helpers_reject_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        solution.positive_weighted_sum([], np.array([]))
    with pytest.raises(ValueError):
        solution.positive_weighted_sum([np.eye(2), np.eye(3)], np.ones(2))
    with pytest.raises(ValueError):
        solution.product_kernel_grams([np.ones((2, 3))])


def test_scale_kernel_gram_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="values"):
        solution.scale_kernel_gram(np.eye(2), np.ones(3))
    with pytest.raises(ValueError, match="values"):
        solution.scale_kernel_gram(np.eye(2), np.array([1.0, np.nan]))


def test_inputs_are_not_modified() -> None:
    X = sample_data()
    X_before = X.copy()
    gram = solution.kernel_matrix(X, X, kernel="rbf", gamma=0.5)
    gram_before = gram.copy()
    values = np.array([-1.0, 0.0, 2.0, 3.0])
    values_before = values.copy()
    solution.finite_feature_coordinates(gram)
    solution.scale_kernel_gram(gram, values)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(gram, gram_before)
    np.testing.assert_array_equal(values, values_before)


def test_guided_demo_runs_and_exposes_sigmoid_counterexample() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "combined PSD: True" in result.stdout
    assert "reconstruction: True" in result.stdout
    assert "sigmoid candidate PSD: False" in result.stdout
    assert "finite sample check is not a global proof" in result.stdout
