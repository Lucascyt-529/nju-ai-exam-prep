import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = (
    ROOT
    / "watermelon_book"
    / "03_linear_models"
    / "04_multiclass_reduction"
    / "reference"
    / "solution.py"
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("multiclass_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def sample_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.5, 0.5],
            [-2.5, -0.5],
            [2.0, 0.0],
            [1.5, 0.5],
            [2.5, -0.5],
            [0.0, 3.0],
            [0.5, 2.5],
            [-0.5, 2.5],
        ]
    )
    y = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30])
    return X, y


def test_ovr_shapes_scores_and_non_contiguous_labels() -> None:
    X, y = sample_data()
    classes, weights, biases = solution.fit_ovr(X, y)
    scores = solution.decision_function_ovr(X, weights, biases)
    np.testing.assert_array_equal(classes, [10, 20, 30])
    assert weights.shape == (3, 2)
    assert biases.shape == (3,)
    assert scores.shape == (9, 3)
    np.testing.assert_array_equal(solution.predict_ovr(X, classes, weights, biases), y)


def test_ovr_equal_scores_choose_first_sorted_class() -> None:
    classes = np.array([10, 20, 30])
    weights = np.zeros((3, 2))
    biases = np.zeros(3)
    prediction = solution.predict_ovr(np.array([[5.0, -2.0]]), classes, weights, biases)
    np.testing.assert_array_equal(prediction, [10])


def test_ovo_builds_every_pair_once_and_predicts_original_labels() -> None:
    X, y = sample_data()
    classes, pairs, weights, biases = solution.fit_ovo(X, y)
    np.testing.assert_array_equal(classes, [10, 20, 30])
    np.testing.assert_array_equal(pairs, [[0, 1], [0, 2], [1, 2]])
    assert weights.shape == (3, 2)
    assert biases.shape == (3,)
    np.testing.assert_array_equal(solution.predict_ovo(X, classes, pairs, weights, biases), y)


def test_ovo_scores_and_votes_have_expected_shapes_and_vote_totals() -> None:
    X, y = sample_data()
    classes, pairs, weights, biases = solution.fit_ovo(X, y)
    scores = solution.decision_function_ovo(X, pairs, weights, biases)
    votes = solution.ovo_vote_counts(scores, pairs, len(classes))
    assert scores.shape == (9, 3)
    assert votes.shape == (9, 3)
    np.testing.assert_array_equal(votes.sum(axis=1), np.full(9, 3))


def test_ovo_vote_tie_chooses_first_sorted_class() -> None:
    classes = np.array([10, 20, 30])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    pair_scores = np.array([[1.0, -1.0, 1.0]])
    votes = solution.ovo_vote_counts(pair_scores, pairs, 3)
    np.testing.assert_array_equal(votes, [[1, 1, 1]])
    prediction = classes[np.argmax(votes, axis=1)]
    np.testing.assert_array_equal(prediction, [10])


def test_zero_pair_score_votes_for_earlier_class() -> None:
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    votes = solution.ovo_vote_counts(np.zeros((1, 3)), pairs, 3)
    np.testing.assert_array_equal(votes, [[2, 1, 0]])


def test_four_classes_create_six_ovo_classifiers() -> None:
    X = np.array([[-3.0], [-2.0], [0.0], [1.0], [3.0], [4.0], [6.0], [7.0]])
    y = np.array([10, 10, 20, 20, 30, 30, 40, 40])
    classes, pairs, weights, biases = solution.fit_ovo(X, y)
    assert classes.shape == (4,)
    assert pairs.shape == (6, 2)
    assert weights.shape == (6, 1)
    assert biases.shape == (6,)


def test_training_does_not_modify_inputs() -> None:
    X, y = sample_data()
    original_X = X.copy()
    original_y = y.copy()
    solution.fit_ovr(X, y)
    solution.fit_ovo(X, y)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.fit_ovr(np.ones((4, 2)), np.array([0, 0, 1, 1])),
        lambda: solution.fit_ovr(np.ones((3, 1)), np.array([0, 1, np.nan])),
        lambda: solution.fit_ovr(
            np.array([[0.0], [0.0], [-1.0], [1.0]]), np.array([0, 0, 1, 2])
        ),
        lambda: solution.predict_ovr(
            np.ones((1, 2)), np.array([0, 1, 2]), np.ones((3, 3)), np.ones(3)
        ),
        lambda: solution.predict_ovr(
            np.ones((1, 2)), np.array([20, 10, 30]), np.ones((3, 2)), np.ones(3)
        ),
        lambda: solution.ovo_vote_counts(
            np.ones((1, 2)), np.array([[0, 1], [0, 2], [1, 2]]), 3
        ),
        lambda: solution.predict_ovo(
            np.ones((1, 2)),
            np.array([10, 20, 30]),
            np.array([[0, 1], [1, 2], [0, 2]]),
            np.ones((3, 2)),
            np.zeros(3),
        ),
    ],
)
def test_invalid_multiclass_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()
