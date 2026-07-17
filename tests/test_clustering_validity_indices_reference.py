import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "09_clustering" / "01_distances_metrics" / "reference" / "solution.py"
spec = importlib.util.spec_from_file_location("validity_solution", SOLUTION)
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def test_jaccard_and_fowlkes_mallows_match_pair_count_formulas():
    truth = np.array([10, 10, 30, 30, 50]); pred = np.array([1, 1, 2, 2, 2])
    assert solution.pair_confusion_counts(truth, pred) == {"same_same": 2, "same_diff": 0, "diff_same": 2, "diff_diff": 6}
    assert solution.jaccard_coefficient(truth, pred) == pytest.approx(.5)
    assert solution.fowlkes_mallows_index(truth, pred) == pytest.approx(np.sqrt(.5))


def test_perfect_nontrivial_partition_has_external_indices_one():
    truth = np.array([0, 0, 1, 1]); pred = np.array([5, 5, 9, 9])
    assert solution.jaccard_coefficient(truth, pred) == pytest.approx(1)
    assert solution.fowlkes_mallows_index(truth, pred) == pytest.approx(1)


def test_all_singletons_make_jaccard_and_fmi_undefined_not_fake_one():
    truth = np.arange(4); pred = np.arange(4)[::-1]
    assert np.isnan(solution.jaccard_coefficient(truth, pred))
    assert np.isnan(solution.fowlkes_mallows_index(truth, pred))


def test_dbi_and_dunn_match_two_one_dimensional_clusters():
    X = np.array([[0.], [2.], [10.], [12.]]); labels = np.array([0, 0, 1, 1])
    assert solution.davies_bouldin_index(X, labels) == pytest.approx(.4)
    assert solution.dunn_index(X, labels) == pytest.approx(4.0)


def test_better_separation_lowers_dbi_and_raises_dunn():
    near = np.array([[0.], [2.], [5.], [7.]]); far = np.array([[0.], [2.], [10.], [12.]])
    labels = np.array([0, 0, 1, 1])
    assert solution.davies_bouldin_index(far, labels) < solution.davies_bouldin_index(near, labels)
    assert solution.dunn_index(far, labels) > solution.dunn_index(near, labels)


def test_singleton_clusters_have_zero_average_distance_and_infinite_dunn_if_separated():
    X = np.array([[0.], [2.], [5.]]); labels = np.array([0, 1, 2])
    assert solution.davies_bouldin_index(X, labels) == pytest.approx(0.0)
    assert np.isinf(solution.dunn_index(X, labels))


def test_equal_centers_make_dbi_infinite():
    X = np.array([[-1.], [1.], [-2.], [2.]]); labels = np.array([0, 0, 1, 1])
    assert np.isinf(solution.davies_bouldin_index(X, labels))


def test_translation_and_label_renaming_leave_internal_indices_unchanged():
    X = np.array([[0., 0.], [0., 2.], [10., 0.], [10., 2.]])
    labels = np.array([0, 0, 1, 1]); renamed = np.array([20, 20, 10, 10])
    assert solution.davies_bouldin_index(X, labels) == pytest.approx(solution.davies_bouldin_index(X + 100, renamed))
    assert solution.dunn_index(X, labels) == pytest.approx(solution.dunn_index(X + 100, renamed))


def test_inputs_are_not_modified():
    X = np.array([[0.], [2.], [10.], [12.]]); labels = np.array([0, 0, 1, 1])
    X_copy, labels_copy = X.copy(), labels.copy()
    solution.davies_bouldin_index(X, labels); solution.dunn_index(X, labels)
    np.testing.assert_array_equal(X, X_copy); np.testing.assert_array_equal(labels, labels_copy)


def test_internal_indices_reject_one_cluster_and_bad_inputs():
    X = np.array([[0.], [1.]])
    with pytest.raises(ValueError): solution.davies_bouldin_index(X, np.array([0, 0]))
    with pytest.raises(ValueError): solution.dunn_index(X, np.array([0, 0]))
    with pytest.raises(ValueError): solution.dunn_index(np.array([0., 1.]), np.array([0, 1]))
    with pytest.raises(ValueError): solution.davies_bouldin_index(X, np.array([0]))
