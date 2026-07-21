import copy
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
    spec = importlib.util.spec_from_file_location("tree_pruning_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def perfect_split_training_data():
    X = np.array([[0], [0], [1], [1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return X, y


def test_prepruning_keeps_split_only_when_validation_accuracy_improves() -> None:
    X, y = perfect_split_training_data()
    tree = solution.fit_prepruned_tree(X, y, X.copy(), y.copy())
    assert not tree["is_leaf"]
    np.testing.assert_array_equal(solution.predict_discrete_tree(tree, X), y)


def test_prepruning_prunes_when_validation_accuracy_is_equal() -> None:
    X, y = perfect_split_training_data()
    X_validation = np.array([[0], [0]], dtype=float)
    y_validation = np.array([0, 1])
    tree = solution.fit_prepruned_tree(X, y, X_validation, y_validation)
    assert tree["is_leaf"]
    assert tree["prediction"] == 0


def test_prepruning_uses_training_data_to_choose_feature() -> None:
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_train = np.array([0, 0, 1, 1])
    X_validation = X_train.copy()
    first = solution.fit_prepruned_tree(
        X_train, y_train, X_validation, np.array([0, 0, 1, 1])
    )
    second = solution.fit_prepruned_tree(
        X_train, y_train, X_validation, np.array([0, 1, 1, 1])
    )
    assert first["feature_index"] == 0
    assert second["feature_index"] == 0


def test_post_pruning_removes_validation_harmful_subtree() -> None:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    full_tree = solution.fit_discrete_tree(X, y)
    pruned = solution.post_prune_tree(full_tree, X, np.zeros(4, dtype=int))
    assert solution.count_tree_nodes(pruned) < solution.count_tree_nodes(full_tree)
    assert pruned["is_leaf"]
    assert pruned["prediction"] == 0


def test_post_pruning_keeps_validation_useful_tree() -> None:
    X, y = perfect_split_training_data()
    full_tree = solution.fit_discrete_tree(X, y)
    pruned = solution.post_prune_tree(full_tree, X, y)
    assert solution.count_tree_nodes(pruned) == solution.count_tree_nodes(full_tree)
    np.testing.assert_array_equal(solution.predict_discrete_tree(pruned, X), y)


def test_post_pruning_uses_conservative_tie_rule() -> None:
    X, y = perfect_split_training_data()
    full_tree = solution.fit_discrete_tree(X, y)
    X_validation = np.array([[0], [0]], dtype=float)
    y_validation = np.array([0, 1])
    pruned = solution.post_prune_tree(full_tree, X_validation, y_validation)
    assert pruned["is_leaf"]


def test_post_pruning_does_not_modify_original_tree() -> None:
    X, y = perfect_split_training_data()
    full_tree = solution.fit_discrete_tree(X, y)
    original = copy.deepcopy(full_tree)
    solution.post_prune_tree(full_tree, X, np.zeros(4, dtype=int))
    assert full_tree == original


def test_validation_unseen_value_uses_existing_fallback_rule() -> None:
    X, y = perfect_split_training_data()
    full_tree = solution.fit_discrete_tree(X, y)
    X_validation = np.array([[2], [2]], dtype=float)
    y_validation = np.array([0, 0])
    pruned = solution.post_prune_tree(full_tree, X_validation, y_validation)
    assert pruned["is_leaf"]
    np.testing.assert_array_equal(
        solution.predict_discrete_tree(pruned, X_validation), [0, 0]
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.fit_prepruned_tree(
            np.ones((2, 1)), np.array([0, 1]), np.ones((2, 2)), np.array([0, 1])
        ),
        lambda: solution.fit_prepruned_tree(
            np.ones((2, 1)), np.array([0, 1]), np.ones((2, 1)), np.array([0])
        ),
        lambda: solution.post_prune_tree({}, np.ones((1, 1)), np.array([0])),
        lambda: solution.post_prune_tree(
            {"is_leaf": True, "prediction": 0, "children": {}},
            np.ones((2, 1)),
            np.array([0]),
        ),
        lambda: solution.count_tree_nodes({}),
    ],
)
def test_invalid_pruning_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError, KeyError)):
        call()
