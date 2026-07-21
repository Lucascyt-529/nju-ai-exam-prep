import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "02_machine_learning" / "00_model_evaluation" / "reference" / "solution.py"
spec = importlib.util.spec_from_file_location("multiclass_metrics_solution", SOLUTION)
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def example():
    return np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0])


def test_per_class_counts_match_hand_confusions():
    y_true, y_pred = example(); report = solution.multiclass_precision_recall_f1(y_true, y_pred)
    np.testing.assert_array_equal(report["labels"], [0, 1, 2])
    np.testing.assert_array_equal(report["tp"], [1, 2, 1])
    np.testing.assert_array_equal(report["fp"], [1, 1, 0])
    np.testing.assert_array_equal(report["fn"], [1, 0, 1])


def test_macro_metrics_use_equal_class_weight_and_harmonic_macro_f1():
    y_true, y_pred = example(); report = solution.multiclass_precision_recall_f1(y_true, y_pred)
    expected_precision = np.array([.5, 2/3, 1.0]); expected_recall = np.array([.5, 1.0, .5])
    np.testing.assert_allclose(report["precision_per_class"], expected_precision)
    np.testing.assert_allclose(report["recall_per_class"], expected_recall)
    macro_p, macro_r = expected_precision.mean(), expected_recall.mean()
    assert report["macro_precision"] == pytest.approx(macro_p)
    assert report["macro_recall"] == pytest.approx(macro_r)
    assert report["macro_f1"] == pytest.approx(2 * macro_p * macro_r / (macro_p + macro_r))


def test_micro_metrics_equal_accuracy_for_single_label_multiclass():
    y_true, y_pred = example(); report = solution.multiclass_precision_recall_f1(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)
    assert report["micro_precision"] == pytest.approx(accuracy)
    assert report["micro_recall"] == pytest.approx(accuracy)
    assert report["micro_f1"] == pytest.approx(accuracy)


def test_absent_requested_class_uses_explicit_zero_denominators():
    report = solution.multiclass_precision_recall_f1(
        np.array([0, 1]), np.array([0, 1]), labels=np.array([0, 1, 2]))
    assert report["precision_per_class"][2] == 0
    assert report["recall_per_class"][2] == 0


def test_string_labels_and_explicit_order_are_supported():
    report = solution.multiclass_precision_recall_f1(
        np.array(["cat", "dog", "bird"]), np.array(["cat", "bird", "bird"]),
        labels=np.array(["dog", "cat", "bird"]))
    np.testing.assert_array_equal(report["labels"], ["dog", "cat", "bird"])


def test_perfect_predictions_give_all_one():
    y = np.array([10, 20, 30, 10]); report = solution.multiclass_precision_recall_f1(y, y.copy())
    for key in ("macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1"):
        assert report[key] == pytest.approx(1.0)


def test_break_even_returns_exact_discrete_intersection_when_present():
    y_true = np.array([1, 0, 1, 0]); scores = np.array([.9, .8, .7, .1])
    # 取前2个时 precision=recall=0.5。
    assert solution.precision_recall_break_even_point(y_true, scores) == pytest.approx(.5)


def test_break_even_interpolates_between_neighboring_pr_points():
    y_true = np.array([1, 0, 1, 0, 1, 0]); scores = np.array([.9, .9, .8, .8, .7, .7])
    precision, recall, _ = solution.precision_recall_curve_points(y_true, scores)
    bep = solution.precision_recall_break_even_point(y_true, scores)
    assert 0 <= bep <= 1
    assert not np.any(np.isclose(precision, recall))


def test_metric_inputs_are_not_modified():
    y_true, y_pred = example(); first, second = y_true.copy(), y_pred.copy()
    solution.multiclass_precision_recall_f1(y_true, y_pred)
    np.testing.assert_array_equal(y_true, first); np.testing.assert_array_equal(y_pred, second)


@pytest.mark.parametrize("true,pred", [
    (np.array([]), np.array([])),
    (np.array([[0, 1]]), np.array([0, 1])),
    (np.array([0, 1]), np.array([0])),
    (np.array([0., np.nan]), np.array([0., 1.])),
])
def test_bad_multiclass_inputs_are_rejected(true, pred):
    with pytest.raises(ValueError): solution.multiclass_precision_recall_f1(true, pred)


def test_bad_explicit_labels_are_rejected():
    y_true, y_pred = example()
    with pytest.raises(ValueError): solution.multiclass_precision_recall_f1(y_true, y_pred, labels=np.array([0, 1]))
    with pytest.raises(ValueError): solution.multiclass_precision_recall_f1(y_true, y_pred, labels=np.array([0, 1, 1, 2]))
