import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = (
    ROOT
    / "02_machine_learning"
    / "06_decision_tree"
    / "reference"
    / "solution.py"
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("discrete_tree_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_entropy_and_gini_are_hand_checkable() -> None:
    balanced = np.array([0, 0, 1, 1])
    pure = np.array([3, 3, 3])
    assert solution.entropy(balanced) == pytest.approx(1.0)
    assert solution.gini(balanced) == pytest.approx(0.5)
    assert solution.entropy(pure) == pytest.approx(0.0)
    assert solution.gini(pure) == pytest.approx(0.0)


def test_majority_label_uses_smallest_label_for_tie() -> None:
    assert solution.majority_label(np.array([20, 10, 20, 10])) == 10


def test_perfect_split_scores_are_hand_checkable() -> None:
    feature = np.array([0, 0, 1, 1])
    y = np.array([10, 10, 20, 20])
    assert solution.information_gain(feature, y) == pytest.approx(1.0)
    assert solution.gain_ratio(feature, y) == pytest.approx(1.0)
    assert solution.gini_index(feature, y) == pytest.approx(0.0)


def test_constant_feature_has_zero_gain_and_ratio() -> None:
    feature = np.zeros(4)
    y = np.array([0, 0, 1, 1])
    assert solution.information_gain(feature, y) == pytest.approx(0.0)
    assert solution.gain_ratio(feature, y) == pytest.approx(0.0)
    assert solution.gini_index(feature, y) == pytest.approx(0.5)


@pytest.mark.parametrize("criterion", ["information_gain", "gain_ratio", "gini"])
def test_all_criteria_choose_perfect_feature_and_fit_training_data(criterion: str) -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([10, 10, 20, 20])
    feature_index, _ = solution.choose_best_feature(
        X, y, np.array([0, 1]), criterion=criterion
    )
    tree = solution.fit_discrete_tree(X, y, criterion=criterion)
    assert feature_index == 0
    np.testing.assert_array_equal(solution.predict_discrete_tree(tree, X), y)


def test_equal_feature_scores_choose_smaller_index() -> None:
    X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    best, _ = solution.choose_best_feature(X, y, np.array([1, 0]))
    assert best == 0


def test_zero_root_gain_can_still_learn_xor_recursively() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    assert solution.information_gain(X[:, 0], y) == pytest.approx(0.0)
    assert solution.information_gain(X[:, 1], y) == pytest.approx(0.0)
    tree = solution.fit_discrete_tree(X, y)
    assert not tree["is_leaf"]
    np.testing.assert_array_equal(solution.predict_discrete_tree(tree, X), y)


def test_unseen_value_falls_back_to_current_node_majority() -> None:
    X = np.array([[0], [0], [1]], dtype=float)
    y = np.array([10, 10, 20])
    tree = solution.fit_discrete_tree(X, y)
    prediction = solution.predict_discrete_tree(tree, np.array([[2]], dtype=float))
    np.testing.assert_array_equal(prediction, [10])


def test_empty_branch_uses_parent_majority_leaf() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 2]], dtype=float)
    y = np.array([0, 1, 1, 1])
    tree = solution.fit_discrete_tree(X, y)
    assert tree["feature_index"] == 0
    left_child = tree["children"][0.0]
    assert left_child["feature_index"] == 1
    empty_branch = left_child["children"][2.0]
    assert empty_branch["is_leaf"]
    assert empty_branch["prediction"] == 0


def test_exhausted_or_constant_features_return_majority_leaf() -> None:
    X = np.ones((4, 2))
    y = np.array([2, 2, 3, 3])
    tree = solution.fit_discrete_tree(X, y)
    assert tree["is_leaf"]
    assert tree["prediction"] == 2


def test_training_does_not_modify_inputs() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    original_X = X.copy()
    original_y = y.copy()
    solution.fit_discrete_tree(X, y)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.entropy(np.array([])),
        lambda: solution.gini(np.array([0.0, np.nan])),
        lambda: solution.information_gain(np.array([0, 1]), np.array([0, 1, 0])),
        lambda: solution.fit_discrete_tree(np.ones((3, 2)), np.array([0, 1])),
        lambda: solution.fit_discrete_tree(
            np.array([[0.0], [np.nan]]), np.array([0, 1])
        ),
        lambda: solution.fit_discrete_tree(
            np.array([[0.0], [1.0]]), np.array([0, 1]), criterion="unknown"
        ),
        lambda: solution.choose_best_feature(
            np.array([[0.0], [1.0]]), np.array([0, 1]), np.array([2])
        ),
        lambda: solution.predict_discrete_tree({}, np.ones((1, 1))),
    ],
)
def test_invalid_discrete_tree_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        call()
