"""学生完成证据：NumPy索引、筛选与同步打乱。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "02_numpy_basics" / "02_indexing_filtering" / "starter.py"


def load_student_module():
    spec = importlib.util.spec_from_file_location("numpy_indexing_student", STARTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


student = load_student_module()


def test_student_selects_tuple_rows_in_requested_order() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    selected = student.select_rows(X, (2, 0))

    np.testing.assert_array_equal(selected, [[3, 30], [1, 10]])
    assert selected.shape == (2, 2)


def test_student_filter_returns_one_dimensional_boolean_mask() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    filtered, mask = student.filter_rows_by_feature(X, 1, 20)

    np.testing.assert_array_equal(mask, [False, True, True])
    np.testing.assert_array_equal(filtered, [[2, 20], [3, 30]])
    assert mask.shape == (3,)
    assert mask.dtype == np.bool_


@pytest.mark.parametrize("target_column", [0, -1])
def test_student_splits_arbitrary_target_column(target_column: int) -> None:
    table = np.array([[1, 10, 100], [2, 20, 200]], dtype=float)

    X, y = student.split_features_target(table, target_column)

    assert X.shape == (2, 2)
    assert y.shape == (2,)
    np.testing.assert_array_equal(y, table[:, target_column])


def test_student_shuffle_is_reproducible_and_keeps_pairs() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)
    y = np.array([100, 200, 300, 400])

    first = student.shuffle_in_unison(X, y, seed=42)
    second = student.shuffle_in_unison(X, y, seed=42)

    np.testing.assert_array_equal(first[2], second[2])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[0][:, 0] * 100, first[1])
    np.testing.assert_array_equal(np.sort(first[2]), np.arange(4))


def test_student_different_seed_changes_this_example_order() -> None:
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10)

    first_order = student.shuffle_in_unison(X, y, seed=1)[2]
    second_order = student.shuffle_in_unison(X, y, seed=2)[2]

    assert not np.array_equal(first_order, second_order)


def test_student_shuffle_rejects_mismatched_sample_counts() -> None:
    with pytest.raises(ValueError, match="行数"):
        student.shuffle_in_unison(
            np.array([[1, 10], [2, 20]], dtype=float),
            np.array([100]),
            seed=0,
        )


def test_student_functions_do_not_modify_inputs() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
    y = np.array([100, 200, 300])
    X_before = X.copy()
    y_before = y.copy()

    student.select_rows(X, [2, 0])
    student.filter_rows_by_feature(X, 1, 20)
    student.shuffle_in_unison(X, y, seed=42)

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
