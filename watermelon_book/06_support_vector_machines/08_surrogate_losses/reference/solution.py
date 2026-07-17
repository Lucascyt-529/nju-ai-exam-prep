"""参考实现：0/1、hinge、指数与对率替代损失的数值比较。"""

import numpy as np


SURROGATE_LOSSES = {"hinge", "exponential", "logistic"}


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or not np.issubdtype(value.dtype, np.number)
        or value.ndim != 1
        or value.size == 0
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(f"{name}必须是非空有限数值一维数组")
    return value.astype(float, copy=False)


def _positive(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name}必须是正有限数值")
    return float(value)


def zero_one_losses(margins: np.ndarray) -> np.ndarray:
    """按教材约定：带符号间隔小于0时记一次错误。"""
    values = _vector(margins, "margins")
    return (values < 0.0).astype(float)


def surrogate_losses(margins: np.ndarray, *, kind: str = "hinge") -> np.ndarray:
    """计算替代损失；logistic使用log2缩放，使边界值等于1。"""
    values = _vector(margins, "margins")
    if kind not in SURROGATE_LOSSES:
        raise ValueError("kind必须是hinge、exponential或logistic")
    if kind == "hinge":
        result = np.maximum(0.0, 1.0 - values)
    elif kind == "exponential":
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.exp(-values)
    else:
        result = np.logaddexp(0.0, -values) / np.log(2.0)
    if not np.all(np.isfinite(result)):
        raise ValueError("损失出现非有限数值；指数损失对极大负间隔容易溢出")
    return result


def margin_gradients(margins: np.ndarray, *, kind: str = "hinge") -> np.ndarray:
    """返回d loss / d margin；hinge在margin=1处取右侧次梯度0。"""
    values = _vector(margins, "margins")
    if kind not in SURROGATE_LOSSES:
        raise ValueError("kind必须是hinge、exponential或logistic")
    if kind == "hinge":
        return np.where(values < 1.0, -1.0, 0.0)
    if kind == "exponential":
        losses = surrogate_losses(values, kind=kind)
        return -losses
    positive = values >= 0.0
    sigmoid_negative = np.empty_like(values)
    sigmoid_negative[positive] = np.exp(-values[positive]) / (
        1.0 + np.exp(-values[positive])
    )
    sigmoid_negative[~positive] = 1.0 / (
        1.0 + np.exp(values[~positive])
    )
    return -sigmoid_negative / np.log(2.0)


def active_gradient_mask(
    margins: np.ndarray, *, kind: str = "hinge", tolerance: float = 1e-12
) -> np.ndarray:
    """标记对当前经验损失仍有非忽略梯度的样本。"""
    tol = _positive(tolerance, "tolerance")
    gradients = margin_gradients(margins, kind=kind)
    return np.abs(gradients) > tol


def regularized_objective(
    weights: np.ndarray,
    margins: np.ndarray,
    C: float,
    *,
    kind: str = "hinge",
) -> dict[str, float]:
    """返回0.5||w||^2、C乘经验损失和总目标。"""
    w = _vector(weights, "weights")
    C_value = _positive(C, "C")
    losses = surrogate_losses(margins, kind=kind)
    regularization = 0.5 * float(w @ w)
    empirical_penalty = C_value * float(np.sum(losses))
    return {
        "regularization": regularization,
        "empirical_penalty": empirical_penalty,
        "objective": regularization + empirical_penalty,
    }


def convexity_gap(
    left: np.ndarray,
    right: np.ndarray,
    mixing: float,
    *,
    kind: str,
) -> np.ndarray:
    """返回loss(theta*x+(1-theta)*y)-加权loss，凸函数应不大于0。"""
    x, y = _vector(left, "left"), _vector(right, "right")
    if x.shape != y.shape:
        raise ValueError("left和right形状必须一致")
    if (
        not isinstance(mixing, (int, float, np.integer, np.floating))
        or isinstance(mixing, (bool, np.bool_))
        or not np.isfinite(mixing)
        or not 0.0 <= mixing <= 1.0
    ):
        raise ValueError("mixing必须位于[0,1]")
    theta = float(mixing)
    mixed = theta * x + (1.0 - theta) * y
    return surrogate_losses(mixed, kind=kind) - (
        theta * surrogate_losses(x, kind=kind)
        + (1.0 - theta) * surrogate_losses(y, kind=kind)
    )
