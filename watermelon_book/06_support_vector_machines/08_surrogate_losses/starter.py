"""学生练习：0/1、hinge、指数与对率替代损失。"""

import numpy as np


def zero_one_losses(margins: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 zero_one_losses")


def surrogate_losses(margins: np.ndarray, *, kind: str = "hinge") -> np.ndarray:
    raise NotImplementedError("请完成 surrogate_losses")


def margin_gradients(margins: np.ndarray, *, kind: str = "hinge") -> np.ndarray:
    raise NotImplementedError("请完成 margin_gradients")


def active_gradient_mask(
    margins: np.ndarray, *, kind: str = "hinge", tolerance: float = 1e-12
) -> np.ndarray:
    raise NotImplementedError("请完成 active_gradient_mask")


def regularized_objective(
    weights: np.ndarray,
    margins: np.ndarray,
    C: float,
    *,
    kind: str = "hinge",
) -> dict[str, float]:
    raise NotImplementedError("请完成 regularized_objective")


def convexity_gap(
    left: np.ndarray,
    right: np.ndarray,
    mixing: float,
    *,
    kind: str,
) -> np.ndarray:
    raise NotImplementedError("请完成 convexity_gap")
