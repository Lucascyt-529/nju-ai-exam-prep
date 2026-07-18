import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "02_numpy_basics" / "03_broadcasting" / "starter.py"


def load_student_module():
    spec = importlib.util.spec_from_file_location("numpy_broadcasting_student", STARTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


student = load_student_module()


def test_feature_offsets_broadcast_across_rows() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    result = student.add_feature_offsets(X, np.array([100, 200], dtype=float))

    np.testing.assert_array_equal(result, [[101, 210], [102, 220], [103, 230]])


def test_sample_offsets_broadcast_across_columns() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    result = student.add_sample_offsets(X, np.array([100, 200, 300], dtype=float))

    np.testing.assert_array_equal(result, [[101, 110], [202, 220], [303, 330]])


def test_center_features_uses_supplied_means() -> None:
    X = np.array([[1, 10], [3, 30]], dtype=float)
    supplied_means = np.array([0, 5], dtype=float)

    result = student.center_features(X, supplied_means)

    np.testing.assert_array_equal(result, [[1, 5], [3, 25]])


def test_safe_standardize_handles_constant_feature_without_mutating_inputs() -> None:
    X = np.array([[1, 10, 5], [3, 30, 5]], dtype=float)
    means = np.array([2, 20, 5], dtype=float)
    stds = np.array([1, 10, 0], dtype=float)
    original_stds = stds.copy()

    standardized, safe_stds = student.safe_standardize(X, means, stds)

    np.testing.assert_array_equal(standardized, [[-1, -1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(safe_stds, [1, 10, 1])
    np.testing.assert_array_equal(stds, original_stds)
    assert not np.shares_memory(safe_stds, stds)


@pytest.mark.parametrize(
    ("function_name", "vector"),
    [
        ("add_feature_offsets", np.array([1, 2, 3], dtype=float)),
        ("add_feature_offsets", np.array([[1, 2]], dtype=float)),
        ("add_sample_offsets", np.array([1, 2], dtype=float)),
        ("add_sample_offsets", np.array([[1, 2, 3]], dtype=float)),
        ("center_features", np.array([1, 2, 3], dtype=float)),
    ],
)
def test_rejects_wrong_vector_shapes(function_name: str, vector: np.ndarray) -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    with pytest.raises(ValueError):
        getattr(student, function_name)(X, vector)


def test_safe_standardize_rejects_wrong_statistics_shapes() -> None:
    X = np.array([[1, 10], [2, 20]], dtype=float)

    with pytest.raises(ValueError):
        student.safe_standardize(X, np.array([1, 2]), np.array([1, 2, 3]))

