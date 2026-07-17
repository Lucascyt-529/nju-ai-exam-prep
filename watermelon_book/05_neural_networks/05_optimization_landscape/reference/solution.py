"""参考实现：一维双井目标的极值与多初值梯度下降。"""

import numpy as np


def _is_finite_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _validate_scalar(value: object, name: str) -> float:
    if not _is_finite_scalar(value):
        raise ValueError(f"{name}必须是有限数值标量")
    return float(value)


def objective(x: float) -> float:
    value = _validate_scalar(x, "x")
    return (value**2 - 1.0) ** 2 + 0.2 * value


def gradient(x: float) -> float:
    value = _validate_scalar(x, "x")
    return 4.0 * value**3 - 4.0 * value + 0.2


def curvature(x: float) -> float:
    value = _validate_scalar(x, "x")
    return 12.0 * value**2 - 4.0


def stationary_points() -> list[float]:
    roots = np.roots([4.0, 0.0, -4.0, 0.2])
    real_roots = sorted(float(root.real) for root in roots if abs(root.imag) < 1e-12)
    if len(real_roots) != 3:
        raise RuntimeError("预期双井目标应有三个实驻点")
    return real_roots


def classify_stationary_point(x: float, *, tolerance: float = 1e-8) -> str:
    point = _validate_scalar(x, "x")
    tolerance_value = _validate_scalar(tolerance, "tolerance")
    if tolerance_value <= 0:
        raise ValueError("tolerance必须为正")
    if abs(gradient(point)) > tolerance_value:
        raise ValueError("给定x不是容差范围内的驻点")
    second_derivative = curvature(point)
    if second_derivative > tolerance_value:
        return "local_minimum"
    if second_derivative < -tolerance_value:
        return "local_maximum"
    return "second_derivative_inconclusive"


def global_minimum_stationary_point() -> tuple[float, float]:
    minima = [
        point
        for point in stationary_points()
        if classify_stationary_point(point) == "local_minimum"
    ]
    point = min(minima, key=objective)
    return point, objective(point)


def gradient_descent(
    start: float,
    *,
    learning_rate: float = 0.05,
    max_steps: int = 500,
    gradient_tolerance: float = 1e-10,
) -> tuple[list[float], list[float]]:
    current = _validate_scalar(start, "start")
    learning_rate_value = _validate_scalar(learning_rate, "learning_rate")
    tolerance_value = _validate_scalar(gradient_tolerance, "gradient_tolerance")
    if learning_rate_value <= 0 or tolerance_value <= 0:
        raise ValueError("learning_rate和gradient_tolerance必须为正")
    if (
        not isinstance(max_steps, (int, np.integer))
        or isinstance(max_steps, (bool, np.bool_))
        or max_steps < 0
    ):
        raise ValueError("max_steps必须是非负整数")

    positions = [current]
    values = [objective(current)]
    for _ in range(int(max_steps)):
        current_gradient = gradient(current)
        if abs(current_gradient) <= tolerance_value:
            break
        current = current - learning_rate_value * current_gradient
        if not np.isfinite(current):
            raise RuntimeError("梯度下降数值发散")
        positions.append(current)
        values.append(objective(current))
    return positions, values


def summarize_starts(
    starts: list[float],
    *,
    learning_rate: float = 0.05,
    max_steps: int = 500,
) -> list[dict[str, float | int]]:
    if not isinstance(starts, list) or not starts:
        raise ValueError("starts必须是非空数值列表")
    original_starts = [_validate_scalar(start, "start") for start in starts]
    summary = []
    for start in original_starts:
        positions, values = gradient_descent(
            start, learning_rate=learning_rate, max_steps=max_steps
        )
        summary.append(
            {
                "start": start,
                "end": positions[-1],
                "value": values[-1],
                "steps": len(positions) - 1,
            }
        )
    return summary
