import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "02_numpy_basics" / "04_matrix_multiplication" / "starter.py"


def load_student_module():
    spec = importlib.util.spec_from_file_location("numpy_matmul_student", STARTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


student = load_student_module()


def test_matrix_product() -> None:
    left = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    right = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    result = student.matrix_product(left, right)

    np.testing.assert_array_equal(result, [[22, 28], [49, 64]])
    assert result.shape == (2, 2)


def test_linear_scores_return_one_score_per_sample() -> None:
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    result = student.linear_scores(X, np.array([10, 1], dtype=float), 5.0)

    np.testing.assert_array_equal(result, [17, 39, 61])
    assert result.shape == (3,)


def test_multi_output_scores_broadcast_bias() -> None:
    X = np.array([[1, 2], [3, 4]], dtype=float)
    W = np.array([[1, 10, 100], [2, 20, 200]], dtype=float)
    b = np.array([5, 50, 500], dtype=float)

    result = student.multi_output_scores(X, W, b)

    np.testing.assert_array_equal(result, [[10, 100, 1000], [16, 160, 1600]])
    assert result.shape == (2, 3)


def test_feature_gram_matrix() -> None:
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

    result = student.feature_gram_matrix(X)

    np.testing.assert_array_equal(result, [[35, 44], [44, 56]])
    assert result.shape == (2, 2)


def test_gradient_shape_matches_weight_shape() -> None:
    X = np.arange(32, dtype=float).reshape(8, 4)
    w = np.ones(4, dtype=float)
    y = np.arange(8, dtype=float)

    error = X @ w - y
    gradient = X.T @ error

    assert error.shape == (8,)
    assert gradient.shape == w.shape == (4,)
