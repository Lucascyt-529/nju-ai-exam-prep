"""运行前先预测形状、首个得分和第一次感知机更新。"""

import numpy as np


def main() -> None:
    X = np.array([[-2.0, -1.0], [2.0, 1.0]])
    y = np.array([-1, 1])
    weights = np.zeros(2)
    bias = 0.0
    learning_rate = 0.5

    first_score = X[0] @ weights + bias
    first_prediction = 1 if first_score >= 0 else -1
    correction = learning_rate * y[0]
    updated_weights = weights + correction * X[0]
    updated_bias = bias + correction

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("weights shape:", weights.shape)
    print("first score:", first_score)
    print("first prediction:", first_prediction)
    print("updated weights:", updated_weights)
    print("updated bias:", updated_bias)


if __name__ == "__main__":
    main()
