"""固定配置训练XOR，观察损失和预测。"""

import numpy as np

from reference.solution import predict_labels, predict_probabilities, train_network


def main() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    parameters, history = train_network(
        X,
        y,
        n_hidden=4,
        learning_rate=1.0,
        epochs=2000,
        seed=0,
    )
    probabilities = predict_probabilities(X, parameters)
    prediction = predict_labels(X, parameters)

    print("X / y:", X.shape, y.shape)
    print("history length:", len(history))
    print("initial loss:", round(history[0], 6))
    print("final loss:", round(history[-1], 6))
    print("probabilities:", np.round(probabilities.ravel(), 4))
    print("prediction:", prediction.ravel())


if __name__ == "__main__":
    main()
