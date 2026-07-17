"""运行前预测单隐层网络各张量形状和广播结果。"""

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def main() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_vector = np.array([0.0, 1.0, 1.0])
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.zeros(2)
    W2 = np.array([[1.0], [-1.0]])
    b2 = np.zeros(1)

    hidden = sigmoid(X @ W1 + b1)
    logits = hidden @ W2 + b2
    probabilities = sigmoid(logits)
    y_column = y_vector.reshape(-1, 1)
    wrong_error = probabilities - y_vector
    correct_error = probabilities - y_column

    print("X:", X.shape)
    print("W1 / b1:", W1.shape, b1.shape)
    print("hidden:", hidden.shape)
    print("W2 / b2:", W2.shape, b2.shape)
    print("logits / probabilities:", logits.shape, probabilities.shape)
    print("y vector / column:", y_vector.shape, y_column.shape)
    print("wrong error:", wrong_error.shape)
    print("correct error:", correct_error.shape)


if __name__ == "__main__":
    main()
