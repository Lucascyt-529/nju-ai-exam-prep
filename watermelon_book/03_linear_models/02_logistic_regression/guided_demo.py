"""运行前预测稳定sigmoid、交叉熵和梯度形状。"""

import numpy as np


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def main() -> None:
    logits = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    print("logits:", logits)
    print("probabilities:", stable_sigmoid(logits))

    X = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 1.0]])
    y = np.array([1.0, 0.0, 1.0])
    w = np.array([0.2, -0.3])
    b = 0.1
    scores = X @ w + b
    errors = stable_sigmoid(scores) - y
    print("score shape:", scores.shape)
    print("error shape:", errors.shape)
    print("gradient_w shape:", (X.T @ errors).shape)


if __name__ == "__main__":
    main()
