import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = (
    ROOT
    / "02_numpy_basics"
    / "02_indexing_filtering"
    / "reference"
    / "solution.py"
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("numpy_indexing_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_select_rows_preserves_requested_order_and_two_dimensions() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    selected = solution.select_rows(X, [2, 0])

    np.testing.assert_array_equal(selected, [[3, 30], [1, 10]])
    assert selected.shape == (2, 2)


def test_filter_returns_rows_and_matching_boolean_mask() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)

    filtered, mask = solution.filter_rows_by_feature(X, 1, 20)

    np.testing.assert_array_equal(mask, [False, True, True])
    np.testing.assert_array_equal(filtered, [[2, 20], [3, 30]])
    assert mask.dtype == np.bool_


@pytest.mark.parametrize("target_column", [0, -1])
def test_split_features_and_target_keeps_expected_shapes(target_column: int) -> None:
    table = np.array([[1, 10, 100], [2, 20, 200]], dtype=float)

    X, y = solution.split_features_target(table, target_column)

    assert X.shape == (2, 2)
    assert y.shape == (2,)
    np.testing.assert_array_equal(y, table[:, target_column])


def test_shuffle_uses_one_reproducible_permutation_for_features_and_labels() -> None:
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)
    y = np.array([100, 200, 300, 400])

    X_first, y_first, order_first = solution.shuffle_in_unison(X, y, seed=42)
    X_second, y_second, order_second = solution.shuffle_in_unison(X, y, seed=42)

    np.testing.assert_array_equal(order_first, order_second)
    np.testing.assert_array_equal(X_first, X_second)
    np.testing.assert_array_equal(y_first, y_second)
    np.testing.assert_array_equal(X_first[:, 0] * 100, y_first)


def test_shuffle_rejects_mismatched_sample_counts() -> None:
    X = np.array([[1, 10], [2, 20]], dtype=float)
    y = np.array([100])

    with pytest.raises(ValueError, match="样本数"):
        solution.shuffle_in_unison(X, y, seed=0)
