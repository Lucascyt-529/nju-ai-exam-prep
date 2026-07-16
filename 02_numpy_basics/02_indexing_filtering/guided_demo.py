"""运行前先预测索引、布尔掩码和同步打乱的结果。"""

import numpy as np


def main() -> None:
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    y = np.array([100, 200, 300])

    row_indices = np.array([2, 0])
    mask = X[:, 1] >= 20

    print("X[[2, 0]] =")
    print(X[row_indices])
    print("selected labels:", y[row_indices])
    print("mask:", mask, "shape:", mask.shape)
    print("filtered X =")
    print(X[mask])
    print("filtered y:", y[mask])

    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(X))
    print("permutation:", permutation)
    print("shuffled first feature:", X[permutation, 0])
    print("shuffled labels:", y[permutation])


if __name__ == "__main__":
    main()
