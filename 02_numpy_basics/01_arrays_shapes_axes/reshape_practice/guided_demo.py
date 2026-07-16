"""运行前预测 reshape、转置和拼接后的形状。"""

import numpy as np


def main() -> None:
    y = np.array([10.0, 20.0, 30.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    print("y:", y, y.shape)
    print("column:")
    print(y[:, None])
    print("column shape:", y[:, None].shape)
    print("X.T:")
    print(X.T)
    print("X.T shape:", X.T.shape)
    print("bias + X:")
    print(np.column_stack((np.ones(X.shape[0]), X)))

    first = np.array([[1.0, 10.0], [2.0, 20.0]])
    second = np.array([[3.0, 30.0], [4.0, 40.0]])
    print("stacked samples:")
    print(np.concatenate((first, second), axis=0))


if __name__ == "__main__":
    main()
