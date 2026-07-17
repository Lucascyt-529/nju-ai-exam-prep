"""学生完成证据：reshape、转置、常数列与批次拼接。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
STARTER = (
    ROOT
    / "02_numpy_basics"
    / "01_arrays_shapes_axes"
    / "reshape_practice"
    / "starter.py"
)


def load_student_module():
    spec = importlib.util.spec_from_file_location("numpy_reshape_student", STARTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


student = load_student_module()


def test_student_vector_column_round_trip() -> None:
    vector = np.array([10.0, 20.0, 30.0])
    column = student.as_column(vector)
    flat = student.as_flat(column)

    assert column.shape == (3, 1)
    assert flat.shape == (3,)
    np.testing.assert_array_equal(column[:, 0], vector)
    np.testing.assert_array_equal(flat, vector)


def test_student_transpose_exchanges_indices_not_just_shape() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    transposed = student.transpose_features(X)

    np.testing.assert_array_equal(transposed, [[1, 3, 5], [2, 4, 6]])
    assert transposed.shape == (2, 3)


def test_student_bias_column_is_on_left_and_input_is_unchanged() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    before = X.copy()

    result = student.add_bias_column(X)

    np.testing.assert_array_equal(result[:, 0], np.ones(3))
    np.testing.assert_array_equal(result[:, 1:], X)
    np.testing.assert_array_equal(X, before)
    assert result.shape == (3, 3)


def test_student_stacks_sample_batches_along_axis_zero() -> None:
    first = np.array([[1.0, 10.0], [2.0, 20.0]])
    second = np.array([[3.0, 30.0], [4.0, 40.0]])

    result = student.stack_sample_batches(first, second)

    np.testing.assert_array_equal(
        result, [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    )
    assert result.shape == (4, 2)


@pytest.mark.parametrize(
    ("function", "argument"),
    [
        (student.as_column, [1.0, 2.0]),
        (student.as_flat, [[1.0], [2.0]]),
        (student.transpose_features, [[1.0, 2.0]]),
        (student.add_bias_column, [[1.0, 2.0]]),
    ],
)
def test_student_rejects_non_numpy_inputs(function, argument) -> None:
    with pytest.raises(TypeError):
        function(argument)


@pytest.mark.parametrize(
    ("function", "argument"),
    [
        (student.as_column, np.ones((1, 2))),
        (student.as_column, np.array([])),
        (student.as_flat, np.ones((1, 2))),
        (student.as_flat, np.empty((0, 1))),
        (student.transpose_features, np.empty((0, 2))),
        (student.add_bias_column, np.empty((2, 0))),
    ],
)
def test_student_rejects_invalid_or_empty_shapes(function, argument) -> None:
    with pytest.raises(ValueError):
        function(argument)


def test_student_rejects_empty_or_mismatched_sample_batches() -> None:
    with pytest.raises(ValueError):
        student.stack_sample_batches(np.empty((0, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        student.stack_sample_batches(np.ones((2, 2)), np.ones((2, 3)))
