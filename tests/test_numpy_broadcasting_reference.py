import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = (
    ROOT / "02_numpy_basics" / "03_broadcasting" / "reference" / "solution.py"
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("numpy_broadcasting_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_feature_offsets_broadcast_across_rows() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    result = solution.add_feature_offsets(X, np.array([100, 200], dtype=float))

    np.testing.assert_array_equal(result, [[101, 210], [102, 220], [103, 230]])


def test_sample_offsets_are_reshaped_to_column_vector() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    result = solution.add_sample_offsets(X, np.array([100, 200, 300], dtype=float))

    np.testing.assert_array_equal(result, [[101, 110], [202, 220], [303, 330]])


def test_centering_and_standardization_use_feature_vectors() -> None:
    X = np.array([[1, 10, 5], [3, 30, 5]], dtype=float)
    means = np.array([2, 20, 5], dtype=float)
    stds = np.array([1, 10, 0], dtype=float)

    centered = solution.center_features(X, means)
    standardized, safe_stds = solution.safe_standardize(X, means, stds)

    np.testing.assert_array_equal(centered, [[-1, -10, 0], [1, 10, 0]])
    np.testing.assert_array_equal(standardized, [[-1, -1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(safe_stds, [1, 10, 1])
    np.testing.assert_array_equal(stds, [1, 10, 0])


@pytest.mark.parametrize(
    "offsets",
    [
        np.array([1, 2, 3], dtype=float),
        np.array([[1, 2]], dtype=float),
    ],
)
def test_feature_offsets_reject_wrong_shapes(offsets: np.ndarray) -> None:
    X = np.array([[1, 10], [2, 20]], dtype=float)

    with pytest.raises(ValueError, match="n_features"):
        solution.add_feature_offsets(X, offsets)
