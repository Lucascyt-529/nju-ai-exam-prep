import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = (
    ROOT
    / "02_numpy_basics"
    / "04_matrix_multiplication"
    / "reference"
    / "solution.py"
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("numpy_matmul_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_matrix_product_matches_hand_calculation() -> None:
    left = np.array([[1, 2], [3, 4]], dtype=float)
    right = np.array([[5, 6], [7, 8]], dtype=float)

    result = solution.matrix_product(left, right)

    np.testing.assert_array_equal(result, [[19, 22], [43, 50]])


def test_linear_scores_return_one_value_per_sample() -> None:
    X = np.array([[1, 2], [3, 4]], dtype=float)
    w = np.array([10, 1], dtype=float)

    result = solution.linear_scores(X, w, b=5.0)

    np.testing.assert_array_equal(result, [17, 39])
    assert result.shape == (2,)


def test_multi_output_scores_use_feature_and_output_dimensions() -> None:
    X = np.array([[1, 2], [3, 4]], dtype=float)
    W = np.array([[1, 10], [2, 20]], dtype=float)
    b = np.array([1, 2], dtype=float)

    result = solution.multi_output_scores(X, W, b)

    np.testing.assert_array_equal(result, [[6, 52], [12, 112]])
    assert result.shape == (2, 2)


def test_gram_matrix_is_symmetric_and_matches_expected_values() -> None:
    X = np.array([[1, 2], [3, 4]], dtype=float)

    gram = solution.feature_gram_matrix(X)

    np.testing.assert_array_equal(gram, [[10, 14], [14, 20]])
    np.testing.assert_array_equal(gram, gram.T)


def test_linear_scores_reject_wrong_weight_shape() -> None:
    X = np.array([[1, 2], [3, 4]], dtype=float)

    with pytest.raises(ValueError, match="n_features"):
        solution.linear_scores(X, np.array([1, 2, 3], dtype=float), b=0.0)
