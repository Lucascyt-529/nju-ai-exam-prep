import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "04_decision_trees" / "03_continuous_mixed_tree" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("continuous_tree_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_candidate_thresholds_use_sorted_distinct_midpoints() -> None:
    np.testing.assert_allclose(solution.candidate_thresholds(np.array([5.0, 1.0, 1.0, 3.0])), [2.0, 4.0])


@pytest.mark.parametrize(
    "criterion, expected",
    [("information_gain", 1.0), ("gain_ratio", 1.0), ("gini", 0.0)],
)
def test_perfect_continuous_split_is_hand_checkable(criterion: str, expected: float) -> None:
    feature = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])
    threshold, score = solution.best_continuous_split(feature, y, criterion=criterion)
    assert threshold == pytest.approx(2.5)
    assert score == pytest.approx(expected)


def test_equal_threshold_scores_choose_smaller_threshold() -> None:
    feature = np.array([1.0, 2.0, 3.0])
    y = np.array([0, 1, 0])
    threshold, _ = solution.best_continuous_split(feature, y)
    assert threshold == pytest.approx(1.5)


@pytest.mark.parametrize("criterion", ["information_gain", "gain_ratio", "gini"])
def test_continuous_feature_can_be_reused_for_three_intervals(criterion: str) -> None:
    X = np.arange(1.0, 7.0).reshape(-1, 1)
    y = np.array([10, 10, 20, 20, 30, 30])
    tree = solution.fit_mixed_tree(X, y, ["continuous"], criterion=criterion)
    np.testing.assert_array_equal(solution.predict_mixed_tree(tree, X), y)
    assert not tree["is_leaf"]
    assert any(not child["is_leaf"] for child in tree["children"].values())


def test_outside_training_range_still_follows_continuous_branches() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = solution.fit_mixed_tree(X, y, ["continuous"])
    np.testing.assert_array_equal(solution.predict_mixed_tree(tree, np.array([[-100.0], [100.0]])), [0, 1])


def test_mixed_tree_uses_discrete_and_continuous_nodes() -> None:
    X = np.array([[0, 1.0], [0, 3.0], [0, 5.0], [1, 1.0], [1, 3.0], [1, 5.0]])
    y = np.array([0, 1, 1, 2, 2, 3])
    tree = solution.fit_mixed_tree(X, y, ["discrete", "continuous"])
    np.testing.assert_array_equal(solution.predict_mixed_tree(tree, X), y)


def test_continuous_gain_ratio_matches_gain_over_binary_intrinsic_value() -> None:
    feature = np.arange(1.0, 6.0)
    y = np.array([0, 0, 1, 1, 1])
    threshold = 2.5
    gain = solution.continuous_split_score(feature, y, threshold, criterion="information_gain")
    ratio = solution.continuous_split_score(feature, y, threshold, criterion="gain_ratio")
    p_left = 2 / 5
    intrinsic = -(p_left * np.log2(p_left) + (1 - p_left) * np.log2(1 - p_left))
    assert ratio == pytest.approx(gain / intrinsic)


def test_gain_ratio_tree_uses_average_gain_filter_before_ratio() -> None:
    # feature 1隔离一个正例，原始增益率更高但信息增益低于候选平均值。
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    feature_high_gain = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    feature_high_ratio_low_gain = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    X = np.column_stack((feature_high_gain, feature_high_ratio_low_gain)).astype(float)
    tree = solution.fit_mixed_tree(X, y, ["discrete", "discrete"], criterion="gain_ratio")
    assert tree["feature_index"] == 0
    assert tree["information_gain"] > 0


def test_unseen_discrete_value_falls_back_to_node_majority() -> None:
    X = np.array([[0.0], [0.0], [1.0]])
    y = np.array([10, 10, 20])
    tree = solution.fit_mixed_tree(X, y, ["discrete"])
    np.testing.assert_array_equal(solution.predict_mixed_tree(tree, np.array([[2.0]])), [10])


def test_constant_features_return_majority_leaf_and_inputs_are_unchanged() -> None:
    X = np.ones((4, 2))
    y = np.array([2, 2, 3, 3])
    original_X, original_y = X.copy(), y.copy()
    tree = solution.fit_mixed_tree(X, y, ["continuous", "discrete"])
    assert tree["is_leaf"] and tree["prediction"] == 2
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.candidate_thresholds(np.array([])),
        lambda: solution.best_continuous_split(np.ones(3), np.array([0, 1, 1])),
        lambda: solution.continuous_split_score(np.arange(3.0), np.array([0, 1, 1]), 10.0),
        lambda: solution.fit_mixed_tree(np.ones((2, 1)), np.array([0, 1]), []),
        lambda: solution.fit_mixed_tree(np.ones((2, 1)), np.array([0, 1]), ["unknown"]),
        lambda: solution.fit_mixed_tree(np.array([[0.0], [np.nan]]), np.array([0, 1]), ["continuous"]),
        lambda: solution.predict_mixed_tree({}, np.ones((1, 1))),
    ],
)
def test_invalid_continuous_tree_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        call()
