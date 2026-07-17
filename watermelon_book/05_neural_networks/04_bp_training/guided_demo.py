"""比较标准BP与累积BP，再用固定配置训练XOR。"""

import numpy as np

from reference.solution import (
    make_epoch_sample_orders,
    predict_labels,
    predict_probabilities,
    train_network_accumulated_bp,
    train_network_standard_bp,
)


def main() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    orders = make_epoch_sample_orders(4, 2, shuffle=True, random_state=7)
    print("standard BP sample orders:", [order.tolist() for order in orders])

    standard_parameters, standard_history = train_network_standard_bp(
        X,
        y,
        n_hidden=4,
        learning_rate=0.5,
        epochs=2,
        seed=0,
        shuffle=True,
        random_state=7,
    )
    accumulated_parameters, accumulated_history = train_network_accumulated_bp(
        X,
        y,
        n_hidden=4,
        learning_rate=0.5,
        epochs=2,
        seed=0,
    )
    print("standard BP losses:", np.round(standard_history, 6))
    print("accumulated BP losses:", np.round(accumulated_history, 6))
    print(
        "same W1 after two epochs:",
        np.allclose(standard_parameters["W1"], accumulated_parameters["W1"]),
    )

    parameters, history = train_network_accumulated_bp(
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
