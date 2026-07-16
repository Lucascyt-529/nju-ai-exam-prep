"""运行前手算线性预测、MSE和梯度形状。"""

import numpy as np


def main() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])
    w = np.array([2.0])
    b = 1.0

    prediction = X @ w + b
    error = prediction - y
    gradient_w = (2.0 / len(X)) * (X.T @ error)
    gradient_b = 2.0 * error.mean()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("w shape:", w.shape)
    print("prediction:", prediction, prediction.shape)
    print("error:", error, error.shape)
    print("MSE:", np.mean(error**2))
    print("gradient_w:", gradient_w, gradient_w.shape)
    print("gradient_b:", gradient_b)


if __name__ == "__main__":
    main()
