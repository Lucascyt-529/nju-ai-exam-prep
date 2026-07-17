"""学生练习：非凸目标的驻点、极值与多初值梯度下降。"""


def objective(x: float) -> float:
    raise NotImplementedError("请完成 objective")


def gradient(x: float) -> float:
    raise NotImplementedError("请完成 gradient")


def curvature(x: float) -> float:
    raise NotImplementedError("请完成 curvature")


def stationary_points() -> list[float]:
    raise NotImplementedError("请完成 stationary_points")


def classify_stationary_point(x: float, *, tolerance: float = 1e-8) -> str:
    raise NotImplementedError("请完成 classify_stationary_point")


def gradient_descent(
    start: float,
    *,
    learning_rate: float = 0.05,
    max_steps: int = 500,
    gradient_tolerance: float = 1e-10,
) -> tuple[list[float], list[float]]:
    raise NotImplementedError("请完成 gradient_descent")
