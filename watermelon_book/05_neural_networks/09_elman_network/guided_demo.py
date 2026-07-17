"""相同当前输入在不同历史下产生不同Elman输出。"""

import numpy as np

from reference.solution import forward_sequence


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


if __name__ == "__main__":
    main()
