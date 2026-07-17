import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "10_dimensionality_reduction" / "01_knn"
spec = importlib.util.spec_from_file_location("knn_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def train_data() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[0.0], [1.0], [3.0], [4.0]]), np.array(["A", "A", "B", "B"])


def test_pairwise_distance_shape_and_hand_values() -> None:
    X_train, _ = train_data()
    distances = solution.pairwise_euclidean(np.array([[0.5], [3.0]]), X_train)
    assert distances.shape == (2, 4)
    np.testing.assert_allclose(distances, [[0.5, 0.5, 2.5, 3.5], [3.0, 2.0, 0.0, 1.0]])


def test_single_query_stays_two_dimensional() -> None:
    X_train, _ = train_data()
    distances = solution.pairwise_euclidean(np.array([[2.0]]), X_train)
    assert distances.shape == (1, 4)


def test_kneighbors_returns_distance_then_index_matrices() -> None:
    X_train, _ = train_data()
    distances, indices = solution.kneighbors(np.array([[0.8], [3.2]]), X_train, 2)
    assert distances.shape == indices.shape == (2, 2)
    np.testing.assert_array_equal(indices, [[1, 0], [2, 3]])
    np.testing.assert_allclose(distances, [[0.2, 0.8], [0.2, 0.8]])


def test_equal_distance_prefers_smaller_training_index() -> None:
    X_train = np.array([[0.0], [2.0], [4.0]])
    _, indices = solution.kneighbors(np.array([[1.0]]), X_train, 2)
    np.testing.assert_array_equal(indices, [[0, 1]])


def test_uniform_classification_majority_vote() -> None:
    X_train, y_train = train_data()
    prediction = solution.predict_classification(np.array([[0.8], [3.2]]), X_train, y_train, 3)
    np.testing.assert_array_equal(prediction, ["A", "B"])


def test_classification_supports_noncontinuous_numeric_labels() -> None:
    X_train, _ = train_data()
    y_train = np.array([10, 10, 30, 30])
    np.testing.assert_array_equal(solution.predict_classification(np.array([[3.2]]), X_train, y_train, 3), [30])


def test_vote_tie_chooses_sorted_earlier_class() -> None:
    X_train = np.array([[0.0], [2.0]])
    y_train = np.array(["z", "a"])
    np.testing.assert_array_equal(solution.predict_classification(np.array([[1.0]]), X_train, y_train, 2), ["a"])


def test_distance_weighting_can_change_uniform_vote() -> None:
    X_train = np.array([[0.0], [2.0], [2.2]])
    y_train = np.array(["A", "B", "B"])
    query = np.array([[0.1]])
    assert solution.predict_classification(query, X_train, y_train, 3, weights="uniform")[0] == "B"
    assert solution.predict_classification(query, X_train, y_train, 3, weights="distance")[0] == "A"


def test_zero_distance_classification_ignores_nonzero_neighbors() -> None:
    X_train = np.array([[1.0], [1.0], [1.1]])
    y_train = np.array(["A", "B", "B"])
    prediction = solution.predict_classification(np.array([[1.0]]), X_train, y_train, 3, weights="distance")
    np.testing.assert_array_equal(prediction, ["A"])


def test_uniform_regression_is_neighbor_mean() -> None:
    X_train, _ = train_data()
    y_train = np.array([0.0, 1.0, 9.0, 16.0])
    prediction = solution.predict_regression(np.array([[0.8]]), X_train, y_train, 2)
    assert prediction[0] == pytest.approx(0.5)


def test_distance_weighted_regression_matches_hand_calculation() -> None:
    X_train = np.array([[0.0], [2.0]])
    y_train = np.array([0.0, 10.0])
    prediction = solution.predict_regression(np.array([[0.5]]), X_train, y_train, 2, weights="distance")
    assert prediction[0] == pytest.approx(2.5)


def test_zero_distance_regression_averages_only_exact_matches() -> None:
    X_train = np.array([[1.0], [1.0], [1.1]])
    y_train = np.array([2.0, 4.0, 100.0])
    prediction = solution.predict_regression(np.array([[1.0]]), X_train, y_train, 3, weights="distance")
    assert prediction[0] == pytest.approx(3.0)


def test_functions_do_not_modify_inputs() -> None:
    X_train, y_train = train_data()
    query = np.array([[1.5]])
    copies = (X_train.copy(), y_train.copy(), query.copy())
    solution.predict_classification(query, X_train, y_train, 3, weights="distance")
    np.testing.assert_array_equal(X_train, copies[0])
    np.testing.assert_array_equal(y_train, copies[1])
    np.testing.assert_array_equal(query, copies[2])


@pytest.mark.parametrize("k", [0, -1, 5, 1.5, True])
def test_invalid_k_is_rejected(k) -> None:
    X_train, _ = train_data()
    with pytest.raises(ValueError):
        solution.kneighbors(np.array([[1.0]]), X_train, k)


@pytest.mark.parametrize("weights", ["weighted", "", None, 1, []])
def test_invalid_weight_mode_is_rejected(weights) -> None:
    X_train, y_train = train_data()
    with pytest.raises(ValueError):
        solution.predict_classification(np.array([[1.0]]), X_train, y_train, 2, weights=weights)


def test_feature_mismatch_and_bad_matrices_are_rejected() -> None:
    X_train, _ = train_data()
    with pytest.raises(ValueError):
        solution.kneighbors(np.array([[1.0, 2.0]]), X_train, 1)
    with pytest.raises(ValueError):
        solution.kneighbors(np.array([1.0]), X_train, 1)
    with pytest.raises(ValueError):
        solution.kneighbors(np.array([[np.nan]]), X_train, 1)


def test_bad_classification_targets_are_rejected() -> None:
    X_train, _ = train_data()
    query = np.array([[1.0]])
    with pytest.raises(ValueError):
        solution.predict_classification(query, X_train, np.array([[0], [0], [1], [1]]), 2)
    with pytest.raises(ValueError):
        solution.predict_classification(query, X_train, np.array([0.0, 0.0, 1.0, np.nan]), 2)


def test_bad_regression_targets_are_rejected() -> None:
    X_train, _ = train_data()
    query = np.array([[1.0]])
    with pytest.raises(ValueError):
        solution.predict_regression(query, X_train, np.array(["0", "1", "9", "16"]), 2)
    with pytest.raises(ValueError):
        solution.predict_regression(query, X_train, np.array([0.0, 1.0, np.inf, 16.0]), 2)


def test_guided_demo_runs_and_reports_shapes_and_predictions() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "distance shape: (2, 4)" in result.stdout
    assert "neighbor indices: [[1, 0, 2], [2, 3, 1]]" in result.stdout
    assert "uniform class: ['A', 'B']" in result.stdout
