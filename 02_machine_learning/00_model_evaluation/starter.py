"""学生练习：回归、二分类、ROC/AUC和代价敏感度量。"""

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = y_pred - y_true
    mae = np.abs(errors)
    return mae.mean()


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = y_pred - y_true
    squared_error = error ** 2
    return squared_error.mean()


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    ave = y_true.mean()
    berror = (y_true - ave) ** 2
    bmse = berror.mean()
    if bmse == 0:
        return 1.0 if mse == 0 else 0.0
    return 1 - mse/bmse


def binary_confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: object = 1
) -> dict[str, int]:
    raise NotImplementedError("请完成 binary_confusion_counts")


def binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: object = 1
) -> dict[str, float]:
    raise NotImplementedError("请完成 binary_classification_metrics")


def cost_sensitive_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    positive_label: object = 1,
) -> float:
    raise NotImplementedError("请完成 cost_sensitive_error")


def roc_curve_points(
    y_true: np.ndarray, scores: np.ndarray, positive_label: object = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回FPR、TPR和从正无穷开始的阈值。"""
    raise NotImplementedError("请完成 roc_curve_points")


def precision_recall_curve_points(
    y_true: np.ndarray, scores: np.ndarray, positive_label: object = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回precision、recall和从正无穷开始的阈值。"""
    raise NotImplementedError("请完成 precision_recall_curve_points")


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("请完成 auc_trapezoid")


def positive_probability_cost(
    positive_probability: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> np.ndarray:
    raise NotImplementedError("请完成 positive_probability_cost")


def cost_curve_lines(
    fpr: np.ndarray, tpr: np.ndarray, probability_costs: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("请完成 cost_curve_lines")


def cost_curve_from_scores(
    y_true: np.ndarray, scores: np.ndarray, probability_costs: np.ndarray,
    positive_label: object = 1,
) -> dict[str, np.ndarray | float]:
    raise NotImplementedError("请完成 cost_curve_from_scores")


def multiclass_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    raise NotImplementedError("请完成 multiclass_precision_recall_f1")


def precision_recall_break_even_point(
    y_true: np.ndarray, scores: np.ndarray, positive_label: object = 1
) -> float:
    raise NotImplementedError("请完成 precision_recall_break_even_point")
