import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "02_machine_learning" / "00_model_evaluation" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("evaluation_metrics_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_regression_metrics_and_negative_r2() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 2.0])
    assert solution.mean_absolute_error(y_true, y_pred) == pytest.approx(1.0)
    assert solution.mean_squared_error(y_true, y_pred) == pytest.approx(5 / 3)
    assert solution.root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(5 / 3))
    assert solution.r2_score(y_true, y_pred) == pytest.approx(-1.5)


def test_constant_target_r2_has_defined_edge_behavior() -> None:
    y_true = np.array([2.0, 2.0, 2.0])
    assert solution.r2_score(y_true, y_true.copy()) == 1.0
    assert solution.r2_score(y_true, np.array([1.0, 2.0, 3.0])) == 0.0


def test_confusion_counts_and_binary_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    counts = solution.binary_confusion_counts(y_true, y_pred)
    metrics = solution.binary_classification_metrics(y_true, y_pred)
    assert counts == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert metrics == pytest.approx(
        {
            "accuracy": 0.75,
            "error_rate": 0.25,
            "precision": 2 / 3,
            "recall": 1.0,
            "specificity": 0.5,
            "f1": 0.8,
        }
    )


def test_zero_denominators_return_zero_not_nan() -> None:
    metrics = solution.binary_classification_metrics(
        np.array([0, 0, 0]), np.array([0, 0, 0])
    )
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_cost_sensitive_error_uses_different_error_costs() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = solution.cost_sensitive_error(y_true, y_pred, 2.0, 5.0)
    assert result == pytest.approx(7 / 4)


def test_roc_curve_and_auc_known_example() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = solution.roc_curve_points(y_true, scores)
    assert np.isinf(thresholds[0])
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    assert solution.auc_trapezoid(fpr, tpr) == pytest.approx(0.75)


def test_precision_recall_curve_endpoints() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall, thresholds = solution.precision_recall_curve_points(
        y_true, scores
    )
    assert np.isinf(thresholds[0])
    assert (precision[0], recall[0]) == (1.0, 0.0)
    assert recall[-1] == 1.0
    assert precision[-1] == 0.5
    assert np.all(np.diff(recall) >= 0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.mean_squared_error(np.array([1.0]), np.array([1.0, 2.0])),
        lambda: solution.mean_absolute_error(np.array([np.nan]), np.array([1.0])),
        lambda: solution.binary_confusion_counts(np.array([0, 1, 2]), np.array([0, 1, 2])),
        lambda: solution.roc_curve_points(np.array([1, 1]), np.array([0.2, 0.8])),
        lambda: solution.auc_trapezoid(np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0, 1.0])),
    ],
)
def test_invalid_metric_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()
