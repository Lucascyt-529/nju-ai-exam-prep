import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "04_decision_trees" / "05_multivariate_tree" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("multivariate_tree_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def oblique_data():
    X = np.array([[-2, 1], [-1, 0], [0, -1], [-1, 2], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_projection_values_are_matrix_vector_product() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(solution.projection_values(X, np.array([2.0, -1.0])), [0.0, 2.0])


def test_oblique_direction_has_pure_split_but_axes_do_not() -> None:
    X, y = oblique_data()
    _, _, axis_gini = solution.best_axis_parallel_split(X, y)
    threshold, oblique_gini = solution.best_projection_threshold(X, y, np.array([1.0, 1.0]))
    assert axis_gini > 0
    assert threshold == pytest.approx(0.0)
    assert oblique_gini == pytest.approx(0.0)


def test_coordinate_search_improves_axis_start() -> None:
    X, y = oblique_data()
    _, _, initial_gini = solution.best_axis_parallel_split(X, y)
    weights, _, searched_gini = solution.coordinate_search_split(
        X, y, np.array([1.0, 0.0]), max_rounds=4
    )
    assert searched_gini < initial_gini
    assert searched_gini == pytest.approx(0.0)
    assert np.count_nonzero(np.abs(weights) > 1e-8) == 2


def test_find_split_and_tree_fit_oblique_data() -> None:
    X, y = oblique_data()
    weights, _, score = solution.find_linear_split(X, y)
    assert score == pytest.approx(0.0)
    assert weights.shape == (2,)
    tree = solution.fit_multivariate_tree(X, y, max_depth=2)
    np.testing.assert_array_equal(solution.predict_multivariate_tree(tree, X), y)
    assert solution.count_tree_nodes(tree) == 3


def test_rotated_data_remains_linearly_separable() -> None:
    X, y = oblique_data()
    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = X @ rotation.T
    _, _, score = solution.find_linear_split(rotated, y)
    assert score == pytest.approx(0.0)


def test_recursive_tree_preserves_non_contiguous_multiclass_labels() -> None:
    X = np.array([[-3, -3], [-2, -2], [0, 0], [1, 1], [3, 3], [4, 4]], dtype=float)
    y = np.array([10, 10, 20, 20, 30, 30])
    tree = solution.fit_multivariate_tree(X, y, max_depth=3)
    np.testing.assert_array_equal(solution.predict_multivariate_tree(tree, X), y)


def test_max_depth_zero_returns_majority_leaf() -> None:
    X, y = oblique_data()
    tree = solution.fit_multivariate_tree(X, y, max_depth=0)
    assert tree["is_leaf"]
    assert tree["prediction"] == 0


def test_min_samples_leaf_is_respected() -> None:
    X, y = oblique_data()
    weights, threshold, _ = solution.find_linear_split(X, y, min_samples_leaf=2)
    left = solution.projection_values(X, weights) <= threshold
    assert left.sum() >= 2 and (~left).sum() >= 2


def test_results_are_deterministic_and_inputs_unchanged() -> None:
    X, y = oblique_data()
    original_X, original_y = X.copy(), y.copy()
    first = solution.find_linear_split(X, y)
    second = solution.find_linear_split(X, y)
    np.testing.assert_allclose(first[0], second[0])
    assert first[1:] == pytest.approx(second[1:])
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.projection_values(np.ones((2, 2)), np.zeros(2)),
        lambda: solution.best_projection_threshold(np.ones((3, 2)), np.array([0, 1, 1]), np.ones(2)),
        lambda: solution.coordinate_search_split(
            np.eye(2), np.array([0, 1]), np.ones(2), initial_step=0.0
        ),
        lambda: solution.fit_multivariate_tree(np.ones((2, 2)), np.array([0]), max_depth=1),
        lambda: solution.fit_multivariate_tree(np.eye(2), np.array([0, 1]), max_depth=-1),
        lambda: solution.fit_multivariate_tree(np.eye(2), np.array([0, 1]), min_samples_leaf=0),
        lambda: solution.predict_multivariate_tree({}, np.ones((1, 2))),
    ],
)
def test_invalid_multivariate_tree_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        call()
