"""固定数据训练一维SOM，观察原型和量化误差。"""

import numpy as np

from reference.solution import map_samples, train_som


def main() -> None:
    X = np.array([[-2.2], [-2.0], [-1.8], [1.8], [2.0], [2.2]])
    prototypes, history = train_som(X, n_neurons=4, epochs=40, seed=3)
    winners = map_samples(X, prototypes)

    print("X:", X.shape)
    print("prototypes:", prototypes.shape, np.round(prototypes.ravel(), 4))
    print("history:", len(history), round(history[0], 6), round(history[-1], 6))
    print("winners:", winners.shape, winners)


if __name__ == "__main__":
    main()
