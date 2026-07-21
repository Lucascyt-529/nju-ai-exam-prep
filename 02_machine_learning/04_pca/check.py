"""核对 PCA 一维重构误差。"""

import numpy as np

import starter


def main() -> int:
    X = np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])
    try:
        model = starter.fit_pca(X, 1)
        Z = starter.transform_pca(X, model["mean"], model["components"])
        reconstructed = starter.inverse_transform_pca(Z, model["mean"], model["components"])
        actual = starter.reconstruction_mse(X, reconstructed)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    correct = np.isclose(actual, 0.5)
    print("重构 MSE 期望: 0.5")
    print("重构 MSE 实际:", actual)
    print("一致:", correct)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
