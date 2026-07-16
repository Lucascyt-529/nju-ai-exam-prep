import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "04_decision_trees" / "04_missing_values" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("missing_tree_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_weighted_entropy_uses_weight_not_raw_count() -> None:
    y = np.array([0, 1])
    assert solution.weighted_entropy(y, np.array([1.0, 1.0])) == pytest.approx(1.0)
    expected = -(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))
    assert solution.weighted_entropy(y, np.array([9.0, 1.0])) == pytest.approx(expected)


def test_branch_plan_distributes_missing_weight_and_conserves_every_row() -> None:
    feature = np.array([0.0, 0.0, 1.0, np.nan])
    weights = np.array([1.0, 1.0, 1.0, 2.0])
    values, proportions, plan = solution.branch_weight_plan(feature, weights)
    np.testing.assert_array_equal(values, [0.0, 1.0])
    np.testing.assert_allclose(proportions, [2 / 3, 1 / 3])
    np.testing.assert_allclose(plan[-1], [4 / 3, 2 / 3])
    np.testing.assert_allclose(plan.sum(axis=1), weights)


def test_zero_weight_known_value_does_not_create_a_branch() -> None:
    feature = np.array([0.0, 1.0, 2.0, np.nan])
    weights = np.array([1.0, 1.0, 0.0, 1.0])
    values, proportions, plan = solution.branch_weight_plan(feature, weights)
    np.testing.assert_array_equal(values, [0.0, 1.0])
    np.testing.assert_allclose(proportions, [0.5, 0.5])
    np.testing.assert_allclose(plan.sum(axis=1), weights)


def test_missing_gain_is_known_gain_times_known_weight_ratio() -> None:
    feature = np.array([0.0, 0.0, 1.0, 1.0, np.nan])
    y = np.array([0, 0, 1, 1, 1])
    assert solution.missing_information_gain(feature, y) == pytest.approx(0.8)


def test_custom_weights_change_effective_known_ratio() -> None:
    feature = np.array([0.0, 0.0, 1.0, 1.0, np.nan])
    y = np.array([0, 0, 1, 1, 1])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 4.0])
    assert solution.missing_information_gain(feature, y, weights) == pytest.approx(0.5)


def test_tree_predicts_known_values_and_preserves_non_contiguous_labels() -> None:
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([10, 10, 20, 20])
    model = solution.fit_missing_value_tree(X, y)
    np.testing.assert_array_equal(solution.predict_missing_tree(model, X), y)


def test_missing_prediction_aggregates_branch_probabilities() -> None:
    X = np.array([[0.0], [0.0], [0.0], [1.0]])
    y = np.array([0, 0, 0, 1])
    model = solution.fit_missing_value_tree(X, y)
    probabilities = solution.predict_proba_missing_tree(model, np.array([[np.nan]]))
    np.testing.assert_allclose(probabilities, [[0.75, 0.25]])
    np.testing.assert_array_equal(solution.predict_missing_tree(model, np.array([[np.nan]])), [0])


def test_unseen_known_value_falls_back_to_node_distribution() -> None:
    X = np.array([[0.0], [0.0], [1.0]])
    y = np.array([0, 0, 1])
    model = solution.fit_missing_value_tree(X, y)
    probabilities = solution.predict_proba_missing_tree(model, np.array([[2.0]]))
    np.testing.assert_allclose(probabilities, [[2 / 3, 1 / 3]])


def test_all_missing_or_constant_features_return_weighted_majority_leaf() -> None:
    X = np.array([[np.nan, 1.0], [np.nan, 1.0], [np.nan, 1.0]])
    y = np.array([0, 1, 1])
    model = solution.fit_missing_value_tree(X, y)
    assert model["root"]["is_leaf"]
    np.testing.assert_array_equal(solution.predict_missing_tree(model, X), [1, 1, 1])


def test_fit_does_not_modify_inputs_or_custom_weights() -> None:
    X = np.array([[0.0], [1.0], [np.nan]])
    y = np.array([0, 1, 1])
    weights = np.array([1.0, 1.0, 2.0])
    copies = X.copy(), y.copy(), weights.copy()
    solution.fit_missing_value_tree(X, y, weights)
    np.testing.assert_array_equal(X, copies[0])
    np.testing.assert_array_equal(y, copies[1])
    np.testing.assert_array_equal(weights, copies[2])


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.weighted_entropy(np.array([0, 1]), np.array([0.0, 0.0])),
        lambda: solution.branch_weight_plan(np.array([np.nan, np.nan]), np.ones(2)),
        lambda: solution.branch_weight_plan(np.array([0.0, 0.0]), np.ones(2)),
        lambda: solution.missing_information_gain(np.array([0.0, np.inf]), np.array([0, 1])),
        lambda: solution.fit_missing_value_tree(np.ones((2, 1)), np.array([0])),
        lambda: solution.fit_missing_value_tree(np.ones((2, 1)), np.array([0, 1]), np.array([1.0, -1.0])),
        lambda: solution.predict_proba_missing_tree({}, np.ones((1, 1))),
    ],
)
def test_invalid_missing_tree_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        call()
