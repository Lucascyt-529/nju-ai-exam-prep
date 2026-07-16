"""运行前先用纸笔推导矩阵乘法的值和形状。"""

import numpy as np


def main() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([10.0, 1.0])

    print("X shape:", X.shape)
    print("w shape:", w.shape)
    print("X * w shape:", (X * w).shape)
    print("X * w =")
    print(X * w)
    print("X @ w shape:", (X @ w).shape)
    print("X @ w:", X @ w)
    print("X.T shape:", X.T.shape)
    print("X.T @ X =")
    print(X.T @ X)


if __name__ == "__main__":
    main()
