"""引导演示：不要把“长得像相似度”误当成合法核函数。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("kernel_validity_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    linear = kernel.kernel_matrix(X, X, kernel="linear")
    laplacian = kernel.kernel_matrix(X, X, kernel="laplacian", gamma=0.7)
    combined = kernel.positive_weighted_sum(
        [linear, laplacian], np.array([0.25, 0.75])
    )
    coordinates = kernel.finite_feature_coordinates(combined)
    sigmoid = kernel.kernel_matrix(
        X, X, kernel="sigmoid", gamma=1.0, coef0=-1.0
    )

    print("combined PSD:", kernel.gram_diagnostics(combined)["positive_semidefinite"])
    print("finite feature shape:", coordinates.shape)
    print("reconstruction:", np.allclose(coordinates @ coordinates.T, combined))
    sigmoid_report = kernel.gram_diagnostics(sigmoid)
    print("sigmoid candidate PSD:", sigmoid_report["positive_semidefinite"])
    print("sigmoid minimum eigenvalue:", round(sigmoid_report["minimum_eigenvalue"], 6))
    print("finite sample check is not a global proof")


if __name__ == "__main__":
    main()
