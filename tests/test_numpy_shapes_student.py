"""学生完成证据：数组形状与axis专题。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "02_numpy_basics" / "01_arrays_shapes_axes" / "starter.py"


def load_student_module():
    spec = importlib.util.spec_from_file_location("numpy_shapes_student", STARTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


student = load_student_module()


def test_student_matrix_creation_and_description() -> None:
    matrix = student.make_feature_matrix([[1, 10, 100], [2, 20, 200]])

    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float64
    assert student.describe_array(matrix) == {
        "shape": (2, 3),
        "ndim": 2,
        "dtype": "float64",
    }


def test_student_axis_means_have_expected_values_and_shapes() -> None:
    matrix = student.make_feature_matrix(
        [[1, 10, 100], [2, 20, 200], [3, 30, 300]]
    )

    feature_result = student.feature_means(matrix)
    sample_result = student.sample_means(matrix)

    np.testing.assert_allclose(feature_result, [2, 20, 200])
    np.testing.assert_allclose(sample_result, [37, 74, 111])
    assert feature_result.shape == (3,)
    assert sample_result.shape == (3,)


def test_student_select_feature_returns_one_dimensional_column() -> None:
    matrix = student.make_feature_matrix([[1, 10], [2, 20], [3, 30]])

    selected = student.select_feature(matrix, 1)

    np.testing.assert_array_equal(selected, [10, 20, 30])
    assert selected.shape == (3,)


@pytest.mark.parametrize(
    "rows",
    [
        [1, 2, 3],
        [],
        [[], []],
        [[1, 2], [3]],
        [[1, np.nan]],
        [[1, np.inf]],
        [[1, -np.inf]],
    ],
)
def test_student_rejects_invalid_feature_matrices(rows) -> None:
    with pytest.raises(ValueError):
        student.make_feature_matrix(rows)
