"""运行前手算类均值、投影方向和阈值。"""

import numpy as np


def main() -> None:
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    mean0 = X[y == 0].mean(axis=0)
    mean1 = X[y == 1].mean(axis=0)
    centered0 = X[y == 0] - mean0
    centered1 = X[y == 1] - mean1
    scatter = centered0.T @ centered0 + centered1.T @ centered1
    weights = np.linalg.pinv(scatter) @ (mean1 - mean0)
    threshold = 0.5 * ((mean0 @ weights) + (mean1 @ weights))
    print("mean0:", mean0)
    print("mean1:", mean1)
    print("within-class scatter:")
    print(scatter)
    print("weights:", weights)
    print("threshold:", threshold)
    print("projections:", X @ weights)


if __name__ == "__main__":
    main()
