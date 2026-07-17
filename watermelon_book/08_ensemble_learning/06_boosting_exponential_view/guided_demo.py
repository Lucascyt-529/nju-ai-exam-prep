"""引导演示：逐轮归一化因子的乘积等于累计指数风险。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("boosting_view_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    view = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(view)

    y = np.array([-1, -1, 1, 1])
    predictions = np.array([[-1, 1, 1, 1], [-1, -1, -1, 1]])
    errors = np.array([0.25, 1.0 / 6.0])
    alphas = np.array([view.optimal_alpha(error) for error in errors])
    trace = view.trace_additive_rounds(y, predictions, alphas)
    print("alphas:", np.round(alphas, 6))
    print("normalizers:", np.round(trace["normalizers"], 6))
    print("risks:", np.round(trace["risks"], 6))
    print("risk equals product Z:", np.allclose(trace["risks"], trace["normalizer_products"]))
    print("final distribution from scores:", np.round(trace["distributions"][-1], 6))


if __name__ == "__main__":
    main()
