"""参考实现：稳定的NumPy对数几率回归。"""

import numpy as np


def _validate_X(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 必须只包含有限数值")


def _validate_y(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y 必须具有形状 (n_samples,)，不能是二维列数组")
    if not np.all(np.isfinite(y)) or not np.all((y == 0) | (y == 1)):
        raise ValueError("y 必须只包含0和1")


def _validate_weights(X: np.ndarray, weights: np.ndarray) -> None:
    if not isinstance(weights, np.ndarray) or weights.ndim != 1 or weights.shape[0] != X.shape[1]:
        raise ValueError("weights 必须具有形状 (n_features,)")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights 必须只包含有限数值")


def _validate_l2(l2: float) -> None:
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 必须是非负有限数值")


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("values 必须只包含有限数值")
    result = np.empty_like(array, dtype=float)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def binary_cross_entropy_from_logits(
    y: np.ndarray, logits: np.ndarray
) -> float:
    labels = np.asarray(y, dtype=float)
    scores = np.asarray(logits, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("y 和 logits 必须是形状一致的非空一维数组")
    if not np.all((labels == 0) | (labels == 1)) or not np.all(np.isfinite(scores)):
        raise ValueError("y 必须为0/1且 logits 必须有限")
    return float(np.mean(np.logaddexp(0.0, scores) - labels * scores))


def predict_proba(
    X: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    _validate_X(X)
    _validate_weights(X, weights)
    if not np.isfinite(bias):
        raise ValueError("bias 必须是有限标量")
    return stable_sigmoid(X @ weights + bias)


def logistic_loss(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    l2: float = 0.0,
) -> float:
    _validate_X(X)
    _validate_y(X, y)
    _validate_weights(X, weights)
    _validate_l2(l2)
    logits = X @ weights + bias
    data_loss = binary_cross_entropy_from_logits(y, logits)
    return data_loss + 0.5 * l2 * float(weights @ weights)


def logistic_gradients(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    l2: float = 0.0,
) -> tuple[np.ndarray, float]:
    _validate_X(X)
    _validate_y(X, y)
    _validate_weights(X, weights)
    _validate_l2(l2)
    errors = predict_proba(X, weights, bias) - y
    gradient_weights = X.T @ errors / X.shape[0] + l2 * weights
    gradient_bias = float(errors.mean())
    return gradient_weights, gradient_bias


def logistic_hessian(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    l2: float = 0.0,
) -> np.ndarray:
    """返回参数顺序为[weights..., bias]的(d+1,d+1) Hessian。"""
    _validate_X(X)
    _validate_y(X, y)
    _validate_weights(X, weights)
    _validate_l2(l2)
    if not np.isfinite(bias):
        raise ValueError("bias 必须是有限标量")
    probabilities = predict_proba(X, weights, bias)
    curvature = probabilities * (1.0 - probabilities)
    design = np.column_stack((X, np.ones(X.shape[0], dtype=float)))
    hessian = design.T @ (design * curvature[:, None]) / X.shape[0]
    hessian[np.arange(X.shape[1]), np.arange(X.shape[1])] += l2
    return hessian


def newton_direction(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    *,
    l2: float = 0.0,
    damping: float = 0.0,
) -> np.ndarray:
    """求解(H+damping*I) direction = gradient。"""
    if (
        not isinstance(damping, (int, float, np.integer, np.floating))
        or isinstance(damping, (bool, np.bool_))
        or not np.isfinite(damping)
        or damping < 0
    ):
        raise ValueError("damping 必须是非负有限数值")
    gradient_weights, gradient_bias = logistic_gradients(
        X, y, weights, bias, l2=l2
    )
    gradient = np.concatenate((gradient_weights, np.array([gradient_bias])))
    hessian = logistic_hessian(X, y, weights, bias, l2=l2)
    damped_hessian = hessian + float(damping) * np.eye(hessian.shape[0])
    try:
        direction = np.linalg.solve(damped_hessian, gradient)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Hessian奇异，需增加damping或L2") from exc
    if not np.all(np.isfinite(direction)):
        raise FloatingPointError("牛顿方向出现非有限数值")
    return direction


def fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_steps: int,
    l2: float = 0.0,
    damping: float = 1e-8,
    step_size: float = 1.0,
    max_backtracking: int = 20,
    initial_weights: np.ndarray | None = None,
    initial_bias: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """使用阻尼Hessian和回溯步长的牛顿法，返回长度n_steps+1历史。"""
    _validate_X(X)
    _validate_y(X, y)
    _validate_l2(l2)
    if (
        not isinstance(n_steps, (int, np.integer))
        or isinstance(n_steps, (bool, np.bool_))
        or n_steps < 1
    ):
        raise ValueError("n_steps 必须是正整数")
    if (
        not isinstance(step_size, (int, float, np.integer, np.floating))
        or isinstance(step_size, (bool, np.bool_))
        or not np.isfinite(step_size)
        or not 0 < step_size <= 1
    ):
        raise ValueError("step_size 必须位于(0,1]")
    if (
        not isinstance(max_backtracking, (int, np.integer))
        or isinstance(max_backtracking, (bool, np.bool_))
        or max_backtracking < 0
    ):
        raise ValueError("max_backtracking 必须是非负整数")
    if not np.isfinite(initial_bias):
        raise ValueError("initial_bias 必须是有限标量")
    if initial_weights is None:
        weights = np.zeros(X.shape[1], dtype=float)
    else:
        _validate_weights(X, initial_weights)
        weights = initial_weights.astype(float, copy=True)
    bias = float(initial_bias)
    losses = np.empty(int(n_steps) + 1, dtype=float)
    losses[0] = logistic_loss(X, y, weights, bias, l2=l2)

    for step in range(1, int(n_steps) + 1):
        direction = newton_direction(
            X, y, weights, bias, l2=l2, damping=damping
        )
        current_loss = losses[step - 1]
        accepted = False
        alpha = float(step_size)
        for _ in range(int(max_backtracking) + 1):
            candidate_weights = weights - alpha * direction[:-1]
            candidate_bias = bias - alpha * float(direction[-1])
            try:
                candidate_loss = logistic_loss(
                    X, y, candidate_weights, candidate_bias, l2=l2
                )
            except (ValueError, FloatingPointError):
                candidate_loss = float("inf")
            if np.isfinite(candidate_loss) and candidate_loss <= current_loss + 1e-15:
                weights = candidate_weights
                bias = candidate_bias
                losses[step] = candidate_loss
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise FloatingPointError("牛顿步回溯后仍无法得到非增损失")
    return weights, bias, losses


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int,
    l2: float = 0.0,
    initial_weights: np.ndarray | None = None,
    initial_bias: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """返回权重、截距和包含初始损失的历史。"""
    _validate_X(X)
    _validate_y(X, y)
    _validate_l2(l2)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必须是正有限数值")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps 必须是正整数")
    if not np.isfinite(initial_bias):
        raise ValueError("initial_bias 必须是有限标量")
    if initial_weights is None:
        weights = np.zeros(X.shape[1], dtype=float)
    else:
        _validate_weights(X, initial_weights)
        weights = initial_weights.astype(float, copy=True)
    bias = float(initial_bias)
    losses = np.empty(n_steps + 1, dtype=float)
    losses[0] = logistic_loss(X, y, weights, bias, l2=l2)
    for step in range(1, n_steps + 1):
        gradient_weights, gradient_bias = logistic_gradients(
            X, y, weights, bias, l2=l2
        )
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        loss = logistic_loss(X, y, weights, bias, l2=l2)
        if not np.isfinite(loss):
            raise FloatingPointError("训练发散：损失变为非有限数值")
        losses[step] = loss
    return weights, bias, losses


def predict_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("probabilities 必须是非空有限一维数组")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities 必须位于 [0, 1]")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold 必须位于 (0, 1)")
    return (values >= threshold).astype(int)
