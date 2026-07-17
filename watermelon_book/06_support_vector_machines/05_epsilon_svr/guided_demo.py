"""引导演示：epsilon管道与线性SVR支持向量。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("epsilon_svr_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    svr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svr)
    X = np.arange(-2.0, 3.0).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0 + np.array([0.0, 0.1, -0.05, 0.05, 0.0])
    model = svr.fit_epsilon_svr(X, y, C=10.0, epsilon=0.1, tolerance=1e-9)
    prediction = svr.decision_function(model, X)
    residual = y - prediction
    print("prediction shape:", prediction.shape)
    print("beta sum:", round(float(np.sum(model["beta"])), 10))
    print("beta:", np.round(model["beta"], 4))
    print("residuals:", np.round(residual, 4))
    print("tube regions:", svr.tube_regions(residual, model["epsilon"]).tolist())
    print("support indices:", svr.support_vector_indices(model).tolist())


if __name__ == "__main__":
    main()
