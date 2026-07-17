"""运行前预测单隐层BP中每个梯度的形状。"""

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def main() -> None:
    X = np.array([[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8]])
    y = np.array([[0.0], [1.0], [0.0]])
    W1 = np.array([[0.1, -0.2], [0.3, 0.4]])
    b1 = np.array([0.05, -0.05])
    W2 = np.array([[0.2], [-0.3]])
    b2 = np.array([0.1])

    a1 = sigmoid(X @ W1 + b1)
    probabilities = sigmoid(a1 @ W2 + b2)
    dz2 = (probabilities - y) / len(X)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1.0 - a1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    print("probabilities / y:", probabilities.shape, y.shape)
    print("dz2:", dz2.shape)
    print("dW2 / db2:", dW2.shape, db2.shape)
    print("da1 / dz1:", da1.shape, dz1.shape)
    print("dW1 / db1:", dW1.shape, db1.shape)


if __name__ == "__main__":
    main()
