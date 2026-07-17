import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "09_clustering" / "01_distances_metrics"
spec = importlib.util.spec_from_file_location("vdm_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def training():
    X = np.array([["red"], ["red"], ["blue"], ["blue"], ["green"], ["green"]])
    labels = np.array([0, 0, 1, 1, 0, 1])
    return X, labels


def test_fit_vdm_matches_hand_conditional_distributions():
    X, labels = training(); model = solution.fit_vdm(X, labels)
    mapping = {value: index for index, value in enumerate(model["feature_values"][0].tolist())}
    np.testing.assert_allclose(model["probabilities"][0][mapping["red"]], [1, 0])
    np.testing.assert_allclose(model["probabilities"][0][mapping["blue"]], [0, 1])
    np.testing.assert_allclose(model["probabilities"][0][mapping["green"]], [.5, .5])


def test_vdm_l1_matches_hand_distances_between_category_distributions():
    X, labels = training(); model = solution.fit_vdm(X, labels)
    query = np.array([["red"], ["blue"], ["green"]])
    distances = solution.pairwise_vdm(query, query, model, p=1)
    np.testing.assert_allclose(distances, [[0, 2, 1], [2, 0, 1], [1, 1, 0]])


def test_vdm_l2_takes_root_after_probability_difference_sum():
    X, labels = training(); model = solution.fit_vdm(X, labels)
    query = np.array([["red"], ["blue"], ["green"]])
    distances = solution.pairwise_vdm(query, query, model, p=2)
    assert distances[0, 1] == pytest.approx(np.sqrt(2))
    assert distances[0, 2] == pytest.approx(np.sqrt(.5))


def test_laplace_smoothing_keeps_rows_normalized_and_nonzero():
    X, labels = training(); model = solution.fit_vdm(X, labels, alpha=1)
    probabilities = model["probabilities"][0]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities > 0)


def test_weighted_minkowski_matches_hand_formula():
    X = np.array([[0., 0.]]); Z = np.array([[3., 4.]])
    assert solution.pairwise_weighted_minkowski(X, Z, p=2, weights=np.array([2., .5]))[0, 0] == pytest.approx(np.sqrt(26))


def test_mixed_distance_combines_numeric_and_vdm_power_sums():
    categories, labels = training(); model = solution.fit_vdm(categories, labels)
    distance = solution.pairwise_mixed_distance(
        np.array([[0.]]), np.array([[3.]]), np.array([["red"]]), np.array([["blue"]]), model, p=2)
    assert distance[0, 0] == pytest.approx(np.sqrt(11))


def test_mixed_distance_applies_separate_attribute_weights():
    categories, labels = training(); model = solution.fit_vdm(categories, labels)
    distance = solution.pairwise_mixed_distance(
        np.array([[0.]]), np.array([[3.]]), np.array([["red"]]), np.array([["blue"]]), model, p=2,
        numeric_weights=np.array([2.]), categorical_weights=np.array([.5]))
    assert distance[0, 0] == pytest.approx(np.sqrt(19))


def test_pairwise_vdm_is_symmetric_zero_diagonal():
    X, labels = training(); model = solution.fit_vdm(X, labels)
    distances = solution.pairwise_vdm(X, X, model, p=2)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0.0)


def test_multiple_categorical_features_add_before_p_root():
    X = np.array([["r", "small"], ["r", "large"], ["b", "small"], ["b", "large"]])
    labels = np.array([0, 0, 1, 1]); model = solution.fit_vdm(X, labels)
    query = np.array([["r", "small"], ["b", "large"]])
    # 第一列红/蓝的L1 VDM为2，第二列small/large分布相同为0。
    assert solution.pairwise_vdm(query[:1], query[1:], model, p=1)[0, 0] == pytest.approx(2)


def test_unknown_category_is_rejected_instead_of_treated_as_ordered_number():
    X, labels = training(); model = solution.fit_vdm(X, labels)
    with pytest.raises(ValueError, match="未见"):
        solution.pairwise_vdm(np.array([["yellow"]]), np.array([["red"]]), model)


def test_fit_and_distance_do_not_modify_inputs():
    X, labels = training(); X_copy = X.copy(); labels_copy = labels.copy()
    model = solution.fit_vdm(X, labels); solution.pairwise_vdm(X, X, model)
    np.testing.assert_array_equal(X, X_copy); np.testing.assert_array_equal(labels, labels_copy)


@pytest.mark.parametrize("alpha", [-1, np.inf, True])
def test_bad_alpha_is_rejected(alpha):
    X, labels = training()
    with pytest.raises(ValueError): solution.fit_vdm(X, labels, alpha=alpha)


@pytest.mark.parametrize("p", [0, .5, np.inf, True])
def test_bad_power_is_rejected(p):
    X, labels = training(); model = solution.fit_vdm(X, labels)
    with pytest.raises(ValueError): solution.pairwise_vdm(X, X, model, p=p)


def test_bad_weights_shapes_values_and_all_zero_are_rejected():
    X, labels = training(); model = solution.fit_vdm(X, labels)
    for weights in (np.array([1., 2.]), np.array([-1.]), np.array([0.]), np.array([np.nan])):
        with pytest.raises(ValueError): solution.pairwise_vdm(X, X, model, weights=weights)


def test_bad_mixed_shapes_and_row_alignment_are_rejected():
    X, labels = training(); model = solution.fit_vdm(X, labels)
    with pytest.raises(ValueError):
        solution.pairwise_mixed_distance(np.ones((2, 1)), np.ones((1, 1)), X[:1], X[:1], model)
    with pytest.raises(ValueError):
        solution.pairwise_mixed_distance(np.ones((1, 2)), np.ones((1, 1)), X[:1], X[:1], model)


def test_guided_demo_reports_vdm_and_mixed_distance():
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "VDM red/blue/green:" in result.stdout
    assert "mixed distance: 3.316625" in result.stdout
