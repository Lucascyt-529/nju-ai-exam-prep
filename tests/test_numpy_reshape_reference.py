import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "02_numpy_basics" / "01_arrays_shapes_axes" / "reshape_practice" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("numpy_reshape_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_vector_column_round_trip() -> None:
    vector = np.array([10.0, 20.0, 30.0])
    column = solution.as_column(vector)
    flat = solution.as_flat(column)

    assert column.shape == (3, 1)
    assert flat.shape == (3,)
    np.testing.assert_array_equal(flat, vector)


def test_transpose_bias_and_sample_stack() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    transposed = solution.transpose_features(X)
    with_bias = solution.add_bias_column(X)
    stacked = solution.stack_sample_batches(X[:1], X[1:])

    assert transposed.shape == (2, 3)
    np.testing.assert_array_equal(transposed, [[1, 3, 5], [2, 4, 6]])
    assert with_bias.shape == (3, 3)
    np.testing.assert_array_equal(with_bias[:, 0], np.ones(3))
    np.testing.assert_array_equal(with_bias[:, 1:], X)
    np.testing.assert_array_equal(stacked, X)


def test_invalid_shape_changes_are_rejected() -> None:
    with pytest.raises(ValueError):
        solution.as_column(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        solution.as_flat(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        solution.stack_sample_batches(
            np.ones((2, 2)), np.ones((2, 3))
        )
