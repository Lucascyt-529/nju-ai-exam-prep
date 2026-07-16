"""运行前先预测每个表达式的值和形状。"""

import numpy as np


def main() -> None:
    X = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ]
    )

    print("X =")
    print(X)
    print("shape:", X.shape)
    print("ndim:", X.ndim)
    print("dtype:", X.dtype)
    print("X[0] shape:", X[0].shape, "value:", X[0])
    print("X[:, 1] shape:", X[:, 1].shape, "value:", X[:, 1])
    print("X[1:3, 0:2] shape:", X[1:3, 0:2].shape)
    print("feature means:", X.mean(axis=0))
    print("sample means:", X.mean(axis=1))


if __name__ == "__main__":
    main()
