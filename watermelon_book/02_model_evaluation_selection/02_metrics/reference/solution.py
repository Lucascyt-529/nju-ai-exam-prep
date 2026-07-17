"""参考实现：回归、二分类、ROC/AUC和代价敏感度量。"""

import numpy as np


def _regression_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    try:
        true = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("回归目标和预测必须是数值") from exc
    if true.ndim != 1 or pred.ndim != 1 or true.size == 0:
        raise ValueError("y_true 和 y_pred 必须是非空一维数组")
    if true.shape != pred.shape:
        raise ValueError("y_true 和 y_pred 形状必须一致")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("回归目标和预测必须只包含有限数值")
    return true, pred


def _classification_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or true.size == 0:
        raise ValueError("y_true 和 y_pred 必须是非空一维数组")
    if true.shape != pred.shape:
        raise ValueError("y_true 和 y_pred 形状必须一致")
    if np.issubdtype(true.dtype, np.number) and not np.all(np.isfinite(true)):
        raise ValueError("y_true 不能包含非有限数值")
    if np.issubdtype(pred.dtype, np.number) and not np.all(np.isfinite(pred)):
        raise ValueError("y_pred 不能包含非有限数值")
    if np.union1d(true, pred).size > 2:
        raise ValueError("本专题只处理二分类标签")
    return true, pred


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _regression_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _regression_arrays(y_true, y_pred)
    return float(np.mean((pred - true) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _regression_arrays(y_true, y_pred)
    residual_sum = float(np.sum((true - pred) ** 2))
    total_sum = float(np.sum((true - true.mean()) ** 2))
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0
    return 1.0 - residual_sum / total_sum


def binary_confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: object = 1
) -> dict[str, int]:
    true, pred = _classification_arrays(y_true, y_pred)
    true_positive = true == positive_label
    pred_positive = pred == positive_label
    return {
        "tp": int(np.count_nonzero(true_positive & pred_positive)),
        "fp": int(np.count_nonzero(~true_positive & pred_positive)),
        "tn": int(np.count_nonzero(~true_positive & ~pred_positive)),
        "fn": int(np.count_nonzero(true_positive & ~pred_positive)),
    }


def binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: object = 1
) -> dict[str, float]:
    counts = binary_confusion_counts(y_true, y_pred, positive_label)
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "accuracy": _safe_ratio(tp + tn, total),
        "error_rate": _safe_ratio(fp + fn, total),
        "precision": precision,
        "recall": recall,
        "specificity": _safe_ratio(tn, tn + fp),
        "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def cost_sensitive_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    positive_label: object = 1,
) -> float:
    if (
        not np.isfinite(false_positive_cost)
        or not np.isfinite(false_negative_cost)
        or false_positive_cost < 0
        or false_negative_cost < 0
    ):
        raise ValueError("错误代价必须是非负有限数值")
    counts = binary_confusion_counts(y_true, y_pred, positive_label)
    n_samples = sum(counts.values())
    total_cost = (
        counts["fp"] * false_positive_cost
        + counts["fn"] * false_negative_cost
    )
    return float(total_cost / n_samples)


def _curve_inputs(
    y_true: np.ndarray, scores: np.ndarray, positive_label: object
) -> tuple[np.ndarray, np.ndarray, int, int]:
    true = np.asarray(y_true)
    score_array = np.asarray(scores, dtype=float)
    if true.ndim != 1 or score_array.ndim != 1 or true.size == 0:
        raise ValueError("y_true 和 scores 必须是非空一维数组")
    if true.shape != score_array.shape:
        raise ValueError("y_true 和 scores 形状必须一致")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores 必须只包含有限数值")
    if np.unique(true).size != 2:
        raise ValueError("曲线要求真实标签同时包含正类和负类")
    positive = true == positive_label
    n_positive = int(np.count_nonzero(positive))
    n_negative = true.size - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("positive_label 必须对应其中一个类别")
    return positive, score_array, n_positive, n_negative


def roc_curve_points(
    y_true: np.ndarray, scores: np.ndarray, positive_label: object = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回FPR、TPR和从正无穷开始的阈值。"""
    positive, score_array, n_positive, n_negative = _curve_inputs(
        y_true, scores, positive_label
    )
    thresholds = np.concatenate(([np.inf], np.sort(np.unique(score_array))[::-1]))
    fpr = np.empty(thresholds.size, dtype=float)
    tpr = np.empty(thresholds.size, dtype=float)
    for index, threshold in enumerate(thresholds):
        predicted_positive = score_array >= threshold
        tp = np.count_nonzero(positive & predicted_positive)
        fp = np.count_nonzero(~positive & predicted_positive)
        tpr[index] = tp / n_positive
        fpr[index] = fp / n_negative
    return fpr, tpr, thresholds


def precision_recall_curve_points(
    y_true: np.ndarray, scores: np.ndarray, positive_label: object = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回precision、recall和从正无穷开始的阈值。"""
    positive, score_array, n_positive, _ = _curve_inputs(
        y_true, scores, positive_label
    )
    thresholds = np.concatenate(([np.inf], np.sort(np.unique(score_array))[::-1]))
    precision = np.empty(thresholds.size, dtype=float)
    recall = np.empty(thresholds.size, dtype=float)
    for index, threshold in enumerate(thresholds):
        predicted_positive = score_array >= threshold
        tp = int(np.count_nonzero(positive & predicted_positive))
        predicted_count = int(np.count_nonzero(predicted_positive))
        precision[index] = 1.0 if predicted_count == 0 else tp / predicted_count
        recall[index] = tp / n_positive
    return precision, recall, thresholds


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    x_values, y_values = _regression_arrays(x, y)
    if x_values.size < 2:
        raise ValueError("计算面积至少需要两个点")
    if np.any(np.diff(x_values) < 0):
        raise ValueError("x 必须单调不减")
    return float(np.trapezoid(y_values, x_values))


def positive_probability_cost(
    positive_probability: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
) -> np.ndarray:
    """把正类基率与两种误判代价映射到[0,1]代价横轴。"""
    probabilities = np.asarray(positive_probability, dtype=float)
    if (probabilities.ndim != 1 or probabilities.size == 0
            or not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("positive_probability必须是[0,1]内的非空有限一维数组")
    if (not np.isfinite(false_positive_cost) or not np.isfinite(false_negative_cost)
            or false_positive_cost < 0 or false_negative_cost < 0
            or false_positive_cost + false_negative_cost == 0):
        raise ValueError("两种错误代价必须非负、有限且不能同时为0")
    numerator = probabilities * false_negative_cost
    denominator = numerator + (1.0 - probabilities) * false_positive_cost
    result = np.empty_like(probabilities)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    result[~valid] = probabilities[~valid]
    return result


def cost_curve_lines(
    fpr: np.ndarray, tpr: np.ndarray, probability_costs: np.ndarray
) -> np.ndarray:
    """每个ROC点对应一条 y=(1-x)FPR+xFNR 的代价线。"""
    fpr_values = np.asarray(fpr, dtype=float)
    tpr_values = np.asarray(tpr, dtype=float)
    x_values = np.asarray(probability_costs, dtype=float)
    if (fpr_values.ndim != 1 or tpr_values.ndim != 1 or fpr_values.size == 0
            or fpr_values.shape != tpr_values.shape
            or not np.all(np.isfinite(fpr_values)) or not np.all(np.isfinite(tpr_values))
            or np.any((fpr_values < 0) | (fpr_values > 1))
            or np.any((tpr_values < 0) | (tpr_values > 1))):
        raise ValueError("fpr和tpr必须是形状相同的[0,1]有限一维数组")
    if (x_values.ndim != 1 or x_values.size == 0 or not np.all(np.isfinite(x_values))
            or np.any((x_values < 0) | (x_values > 1)) or np.any(np.diff(x_values) < 0)):
        raise ValueError("probability_costs必须是[0,1]内单调不减的非空有限一维数组")
    false_negative_rates = 1.0 - tpr_values
    return fpr_values[:, None] * (1.0 - x_values[None, :]) + false_negative_rates[:, None] * x_values[None, :]


def cost_curve_from_scores(
    y_true: np.ndarray, scores: np.ndarray, probability_costs: np.ndarray,
    positive_label: object = 1,
) -> dict[str, np.ndarray | float]:
    """将所有ROC阈值转成代价线，并返回逐点下包络及其面积。"""
    x_values = np.asarray(probability_costs, dtype=float)
    fpr, tpr, thresholds = roc_curve_points(y_true, scores, positive_label)
    lines = cost_curve_lines(fpr, tpr, x_values)
    envelope = np.min(lines, axis=0)
    area = auc_trapezoid(x_values, envelope) if len(x_values) >= 2 else 0.0
    return {"probability_costs": x_values.copy(), "normalized_cost_lines": lines,
            "lower_envelope": envelope, "area": float(area),
            "fpr": fpr, "tpr": tpr, "thresholds": thresholds}
