import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "07_bayesian_classifiers" / "07_tan"
spec = importlib.util.spec_from_file_location("tan_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def xor_dependency(repeats=20):
    X = np.tile(np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=int), (repeats, 1))
    y = np.tile(np.array([0, 0, 1, 1], dtype=int), repeats)
    return X, y


def test_conditional_mutual_information_is_log_two_for_deterministic_relation_within_each_class():
    X, y = xor_dependency()
    assert solution.conditional_mutual_information(X, y, 0, 1) == pytest.approx(np.log(2.0))


def test_conditionally_independent_features_have_zero_information():
    rows = []; labels = []
    for label in (0, 1):
        for left in (0, 1):
            for right in (0, 1):
                rows.append([left, right]); labels.append(label)
    assert solution.conditional_mutual_information(np.array(rows), np.array(labels), 0, 1) == pytest.approx(0.0)


def test_conditional_mi_matrix_is_symmetric_zero_diagonal():
    X, y = xor_dependency(); matrix = solution.conditional_mi_matrix(X, y)
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0)


def test_maximum_spanning_tree_uses_largest_edges_and_requested_root():
    weights = np.array([[0, 5, 1, 1], [5, 0, 4, 2], [1, 4, 0, 3], [1, 2, 3, 0]], dtype=float)
    np.testing.assert_array_equal(solution.maximum_spanning_tree(weights, root=0), [-1, 0, 1, 2])


def test_maximum_spanning_tree_ties_are_deterministic():
    weights = np.ones((3, 3)) - np.eye(3)
    np.testing.assert_array_equal(solution.maximum_spanning_tree(weights, root=0), [-1, 0, 0])


def test_fit_tan_shapes_probabilities_and_tree():
    X, y = xor_dependency(); model = solution.fit_tan(X, y, root=0, alpha=1.0)
    np.testing.assert_array_equal(model["parents"], [-1, 0])
    assert model["weights"].shape == (2, 2)
    assert model["class_log_prior"].shape == (2,)
    assert model["log_probabilities"][0].shape == (2, 2)
    assert model["log_probabilities"][1].shape == (2, 2, 2)
    np.testing.assert_allclose(np.exp(model["class_log_prior"]).sum(), 1.0)
    np.testing.assert_allclose(np.exp(model["log_probabilities"][0]).sum(axis=1), 1.0)
    np.testing.assert_allclose(np.exp(model["log_probabilities"][1]).sum(axis=2), 1.0)


def test_tan_classifies_dependency_pattern_that_single_feature_marginals_cannot():
    X, y = xor_dependency(); model = solution.fit_tan(X, y)
    np.testing.assert_array_equal(solution.predict_tan(model, X[:4]), y[:4])
    assert np.mean(solution.predict_tan(model, X) == y) == pytest.approx(1.0)


def test_log_scores_are_finite_and_argmax_matches_prediction():
    X, y = xor_dependency(); model = solution.fit_tan(X, y)
    scores = solution.tan_log_scores(model, X[:4])
    assert scores.shape == (4, 2) and np.all(np.isfinite(scores))
    np.testing.assert_array_equal(model["classes"][np.argmax(scores, axis=1)], solution.predict_tan(model, X[:4]))


def test_noncontiguous_feature_values_and_labels_are_preserved():
    X, y = xor_dependency(); X = X * 10 + 5; y = y * 10 + 10
    model = solution.fit_tan(X, y)
    np.testing.assert_array_equal(model["classes"], [10, 20])
    np.testing.assert_array_equal(solution.predict_tan(model, X[:4]), y[:4])


def test_changing_root_reorients_same_single_tree_edge():
    X, y = xor_dependency(); model = solution.fit_tan(X, y, root=1)
    np.testing.assert_array_equal(model["parents"], [1, -1])
    np.testing.assert_array_equal(solution.predict_tan(model, X[:4]), y[:4])


def test_fit_and_predict_do_not_modify_inputs():
    X, y = xor_dependency(); X_copy = X.copy(); y_copy = y.copy()
    model = solution.fit_tan(X, y); query = X[:4].copy(); query_copy = query.copy()
    solution.predict_tan(model, query)
    np.testing.assert_array_equal(X, X_copy); np.testing.assert_array_equal(y, y_copy)
    np.testing.assert_array_equal(query, query_copy)


def test_unknown_prediction_value_is_rejected_explicitly():
    X, y = xor_dependency(); model = solution.fit_tan(X, y)
    with pytest.raises(ValueError, match="未见"):
        solution.predict_tan(model, np.array([[2, 0]], dtype=int))


@pytest.mark.parametrize("bad_X,bad_y", [
    (np.array([0, 1]), np.array([0, 1])),
    (np.array([[0.0], [1.0]]), np.array([0, 1])),
    (np.array([[0], [1]]), np.array([[0], [1]])),
    (np.array([[0], [1]]), np.array([0, 0])),
])
def test_bad_training_inputs_are_rejected(bad_X, bad_y):
    with pytest.raises(ValueError): solution.fit_tan(bad_X, bad_y)


def test_bad_feature_indices_alpha_and_root_are_rejected():
    X, y = xor_dependency()
    with pytest.raises(ValueError): solution.conditional_mutual_information(X, y, 0, 0)
    with pytest.raises(ValueError): solution.conditional_mutual_information(X, y, 0, 2)
    with pytest.raises(ValueError): solution.fit_tan(X, y, alpha=0)
    with pytest.raises(ValueError): solution.fit_tan(X, y, root=2)


@pytest.mark.parametrize("weights", [np.ones((2, 3)), np.array([[0, 1], [2, 0]]), np.array([[1, 0], [0, 0]]), np.array([[0, np.nan], [np.nan, 0]])])
def test_bad_weight_matrices_are_rejected(weights):
    with pytest.raises(ValueError): solution.maximum_spanning_tree(weights)


def test_guided_demo_runs_and_reports_dependency_success():
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "tree parents: [-1, 0]" in result.stdout
    assert "score shape: (4, 2)" in result.stdout
    assert "predictions: [0, 0, 1, 1]" in result.stdout
    assert "training accuracy: 1.0" in result.stdout
