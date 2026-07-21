"""展示二维数据压缩到一维后的重构误差。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])
    print("输入 X:")
    print(X)
    print("期望一维重构 MSE: 0.5")
    try:
        model = starter.fit_pca(X, 1)
        Z = starter.transform_pca(X, model["mean"], model["components"])
        reconstructed = starter.inverse_transform_pca(Z, model["mean"], model["components"])
        actual = starter.reconstruction_mse(X, reconstructed)
    except NotImplementedError as error:
        print("实际结果:", error)
        return
    print("实际一维重构 MSE:", actual)
    print("一致:", np.isclose(actual, 0.5))


if __name__ == "__main__":
    main()
