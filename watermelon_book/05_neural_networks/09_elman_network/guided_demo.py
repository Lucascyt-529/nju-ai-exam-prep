"""相同当前输入在不同历史下产生不同Elman输出。"""

import numpy as np

from reference.solution import forward_sequence, gradient_check_elman, train_elman_sequence


def main() -> None:
    parameters = {
        "Wx": np.array([[1.0]]),
        "Wh": np.array([[0.8]]),
        "bh": np.zeros(1),
        "Wy": np.array([[1.0]]),
        "by": np.zeros(1),
    }
    positive_history = np.array([[1.0], [0.0]])
    negative_history = np.array([[-1.0], [0.0]])
    positive = forward_sequence(positive_history, parameters)
    negative = forward_sequence(negative_history, parameters)

    print("X:", positive_history.shape)
    print("states / outputs:", positive["states"].shape, positive["outputs"].shape)
    print("same final input:", positive_history[-1], negative_history[-1])
    print("positive-history final output:", np.round(positive["outputs"][-1], 6))
    print("negative-history final output:", np.round(negative["outputs"][-1], 6))
    targets = np.array([[0.5], [-0.25]])
    check = gradient_check_elman(positive_history, targets, parameters)
    trained = train_elman_sequence(positive_history, targets, 2, learning_rate=0.05, epochs=200, seed=3)
    print("BPTT max relative error:", max(check.values()))
    print("training loss first/last:", trained["loss_history"][0], trained["loss_history"][-1])


if __name__ == "__main__":
    main()
