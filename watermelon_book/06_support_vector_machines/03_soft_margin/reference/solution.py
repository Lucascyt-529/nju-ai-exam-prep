"""参考实现：软间隔SVM的损失、样本区域、alpha状态与KKT分析。"""

import numpy as np


OUTSIDE_MARGIN = "outside_margin"
ON_MARGIN = "on_margin"
INSIDE_MARGIN = "inside_margin"
DECISION_BOUNDARY = "decision_boundary"
MISCLASSIFIED = "misclassified"

NON_SUPPORT = "non_support"
FREE_SUPPORT = "free_support"
BOUNDED_SUPPORT = "bounded_support"


def _is_finite_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and np.all(np.isfinite(value))
    )


def _positive_scalar(value: object, name: str) -> float:
    valid = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value > 0
    )
    if not valid:
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def _validate_vector(value: np.ndarray, name: str, *, nonempty: bool = True) -> None:
    if (
        not _is_finite_numeric_array(value)
        or value.ndim != 1
        or (nonempty and value.size == 0)
    ):
        raise ValueError(f"{name}必须是有限数值一维数组")


def _margin_tolerance(value: object) -> float:
    tolerance = _positive_scalar(value, "tolerance")
    if tolerance >= 0.5:
        raise ValueError("tolerance必须小于0.5，避免间隔区域重叠")
    return tolerance


def _alpha_tolerance(value: object, C: float) -> float:
    tolerance = _positive_scalar(value, "tolerance")
    if tolerance >= C / 2.0:
        raise ValueError("tolerance必须小于C/2，避免alpha状态重叠")
    return tolerance


def signed_margins(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    _validate_vector(y, "y")
    _validate_vector(scores, "scores")
    if y.shape != scores.shape or not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y和scores必须同为形状(n,)，且y只能包含-1和+1")
    return y.astype(float, copy=False) * scores.astype(float, copy=False)


def hinge_losses(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - signed_margins(y, scores))


def margin_region_labels(
    margins: np.ndarray, *, tolerance: float = 1e-7
) -> np.ndarray:
    _validate_vector(margins, "margins")
    tol = _margin_tolerance(tolerance)
    labels = np.empty(margins.shape, dtype="<U17")
    labels[margins > 1.0 + tol] = OUTSIDE_MARGIN
    labels[(margins >= 1.0 - tol) & (margins <= 1.0 + tol)] = ON_MARGIN
    labels[(margins > tol) & (margins < 1.0 - tol)] = INSIDE_MARGIN
    labels[(margins >= -tol) & (margins <= tol)] = DECISION_BOUNDARY
    labels[margins < -tol] = MISCLASSIFIED
    return labels


def alpha_status_labels(
    alphas: np.ndarray, C: float, *, tolerance: float = 1e-7
) -> np.ndarray:
    _validate_vector(alphas, "alphas")
    C_value = _positive_scalar(C, "C")
    tol = _alpha_tolerance(tolerance, C_value)
    if np.any(alphas < -tol) or np.any(alphas > C_value + tol):
        raise ValueError("alphas必须位于[0,C]")
    labels = np.full(alphas.shape, FREE_SUPPORT, dtype="<U15")
    labels[alphas <= tol] = NON_SUPPORT
    labels[alphas >= C_value - tol] = BOUNDED_SUPPORT
    return labels


def kkt_consistency_flags(
    margins: np.ndarray,
    alphas: np.ndarray,
    C: float,
    *,
    tolerance: float = 1e-7,
) -> np.ndarray:
    _validate_vector(margins, "margins")
    if margins.shape != alphas.shape:
        raise ValueError("margins和alphas形状必须一致")
    C_value = _positive_scalar(C, "C")
    tol = _margin_tolerance(tolerance)
    _alpha_tolerance(tol, C_value)
    status = alpha_status_labels(alphas, C_value, tolerance=tol)
    flags = np.empty(margins.shape, dtype=bool)
    non_support = status == NON_SUPPORT
    free = status == FREE_SUPPORT
    bounded = status == BOUNDED_SUPPORT
    flags[non_support] = margins[non_support] >= 1.0 - tol
    flags[free] = np.abs(margins[free] - 1.0) <= tol
    flags[bounded] = margins[bounded] <= 1.0 + tol
    return flags


def analyze_soft_margin_solution(
    weights: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    alphas: np.ndarray,
    C: float,
    *,
    tolerance: float = 1e-7,
) -> dict[str, object]:
    _validate_vector(weights, "weights")
    _validate_vector(alphas, "alphas")
    C_value = _positive_scalar(C, "C")
    tol = _margin_tolerance(tolerance)
    _alpha_tolerance(tol, C_value)
    margins = signed_margins(y, scores)
    if alphas.shape != margins.shape:
        raise ValueError("alphas必须与样本数量一致")
    losses = np.maximum(0.0, 1.0 - margins)
    regularization = 0.5 * float(weights.astype(float) @ weights.astype(float))
    slack_penalty = C_value * float(np.sum(losses))
    regions = margin_region_labels(margins, tolerance=tol)
    alpha_status = alpha_status_labels(alphas, C_value, tolerance=tol)
    kkt_ok = kkt_consistency_flags(
        margins, alphas, C_value, tolerance=tol
    )
    return {
        "margins": margins,
        "slacks": losses,
        "hinge_losses": losses.copy(),
        "regions": regions,
        "alpha_status": alpha_status,
        "kkt_ok": kkt_ok,
        "regularization": regularization,
        "slack_penalty": slack_penalty,
        "objective": regularization + slack_penalty,
        "support_count": int(np.count_nonzero(alphas > tol)),
        "margin_violation_count": int(np.count_nonzero(losses > tol)),
        "misclassified_count": int(np.count_nonzero(margins < -tol)),
    }
