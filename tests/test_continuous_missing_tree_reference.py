import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "04_decision_trees" / "04_missing_values"
spec = importlib.util.spec_from_file_location("continuous_missing_tree_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def test_continuous_missing_gain_is_known_gain_times_known_ratio() -> None:
    feature = np.array([1., 2., 3., 4., np.nan]); y = np.array([0, 0, 1, 1, 1])
    assert solution.continuous_missing_information_gain(feature, y, 2.5) == pytest.approx(0.8)


def test_continuous_missing_gain_respects_custom_weight_ratio() -> None:
    feature = np.array([1., 2., 3., 4., np.nan]); y = np.array([0, 0, 1, 1, 1])
    weights = np.array([1., 1., 1., 1., 4.])
    assert solution.continuous_missing_information_gain(feature, y, 2.5, weights) == pytest.approx(0.5)


def test_best_threshold_uses_only_distinct_known_values() -> None:
    feature = np.array([1., 2., np.nan, 3., 4.]); y = np.array([0, 0, 1, 1, 1])
    threshold, gain = solution.best_continuous_missing_split(feature, y)
    assert threshold == pytest.approx(2.5) and gain == pytest.approx(0.8)


def test_continuous_branch_plan_distributes_missing_and_conserves_weight() -> None:
    feature = np.array([1., 2., 4., np.nan]); weights = np.array([1., 1., 1., 2.])
    proportions, plan = solution.continuous_branch_weight_plan(feature, weights, 2.5)
    np.testing.assert_allclose(proportions, [2 / 3, 1 / 3])
    np.testing.assert_allclose(plan[-1], [4 / 3, 2 / 3])
    np.testing.assert_allclose(plan.sum(axis=1), weights)


def test_continuous_tree_predicts_known_values_and_outside_range() -> None:
    X = np.array([[1.], [2.], [3.], [4.]]); y = np.array([10, 10, 20, 20])
    model = solution.fit_missing_value_tree(X, y, feature_types=["continuous"])
    assert model["root"]["split_type"] == "continuous"
    assert model["root"]["threshold"] == pytest.approx(2.5)
    np.testing.assert_array_equal(solution.predict_missing_tree(model, np.array([[-100.], [100.]])), [10, 20])


def test_continuous_feature_remains_available_for_three_intervals() -> None:
    X = np.arange(1., 7.).reshape(-1, 1); y = np.array([10, 10, 20, 20, 30, 30])
    model = solution.fit_missing_value_tree(X, y, feature_types=["continuous"])
    np.testing.assert_array_equal(solution.predict_missing_tree(model, X), y)
    assert any(not child["is_leaf"] for child in model["root"]["children"].values())


def test_missing_continuous_prediction_aggregates_left_right_probabilities() -> None:
    X = np.array([[0.], [0.1], [0.2], [1.]]); y = np.array([0, 0, 0, 1])
    model = solution.fit_missing_value_tree(X, y, feature_types=["continuous"])
    probability = solution.predict_proba_missing_tree(model, np.array([[np.nan]]))
    np.testing.assert_allclose(probability, [[0.75, 0.25]])


def test_missing_training_sample_is_propagated_to_both_continuous_branches() -> None:
    X = np.array([[1.], [2.], [3.], [4.], [np.nan]])
    y = np.array([0, 0, 1, 1, 1]); weights = np.array([1., 1., 1., 1., 2.])
    model = solution.fit_missing_value_tree(X, y, weights, feature_types=["continuous"])
    root = model["root"]
    np.testing.assert_allclose(root["branch_probabilities"], [0.5, 0.5])
    assert set(root["children"]) == {"left", "right"}
    np.testing.assert_allclose(root["children"]["left"]["class_probabilities"], [2 / 3, 1 / 3])
    np.testing.assert_allclose(root["children"]["right"]["class_probabilities"], [0., 1.])


def test_mixed_discrete_and_continuous_missing_tree_fits_training_data() -> None:
    X = np.array([[0., 1.], [0., 3.], [0., 5.], [1., 1.], [1., 3.], [1., 5.]])
    y = np.array([0, 1, 1, 2, 2, 3])
    model = solution.fit_missing_value_tree(X, y, feature_types=["discrete", "continuous"])
    np.testing.assert_array_equal(solution.predict_missing_tree(model, X), y)


def test_default_feature_type_remains_discrete_for_backward_compatibility() -> None:
    X = np.array([[0.], [0.], [1.], [1.]]); y = np.array([0, 0, 1, 1])
    model = solution.fit_missing_value_tree(X, y)
    assert model["feature_types"] == ("discrete",)
    assert model["root"]["split_type"] == "discrete"


def test_continuous_fit_does_not_modify_inputs() -> None:
    X = np.array([[1.], [2.], [3.], [np.nan]]); y = np.array([0, 0, 1, 1]); weights = np.array([1., 1., 1., 2.])
    copies = X.copy(), y.copy(), weights.copy()
    solution.fit_missing_value_tree(X, y, weights, feature_types=["continuous"])
    np.testing.assert_array_equal(X, copies[0]); np.testing.assert_array_equal(y, copies[1]); np.testing.assert_array_equal(weights, copies[2])


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.continuous_missing_information_gain(np.array([1., np.nan]), np.array([0, 1]), 1.0),
        lambda: solution.continuous_branch_weight_plan(np.array([1., 2.]), np.ones(2), 10.0),
        lambda: solution.best_continuous_missing_split(np.array([1., 1., np.nan]), np.array([0, 1, 1])),
        lambda: solution.fit_missing_value_tree(np.ones((2, 1)), np.array([0, 1]), feature_types=[]),
        lambda: solution.fit_missing_value_tree(np.ones((2, 1)), np.array([0, 1]), feature_types=["unknown"]),
    ],
)
def test_invalid_continuous_missing_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_starter_keeps_continuous_missing_tasks_unimplemented() -> None:
    starter_spec = importlib.util.spec_from_file_location("continuous_missing_starter", TOPIC / "starter.py")
    assert starter_spec is not None and starter_spec.loader is not None
    starter = importlib.util.module_from_spec(starter_spec); starter_spec.loader.exec_module(starter)
    with pytest.raises(NotImplementedError):
        starter.best_continuous_missing_split(np.array([1., 2.]), np.array([0, 1]))
