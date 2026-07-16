"""运行前先手算根节点两个特征的信息增益。"""

import numpy as np


def entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def information_gain(feature: np.ndarray, y: np.ndarray) -> float:
    remainder = 0.0
    for value in np.unique(feature):
        selected = feature == value
        remainder += selected.mean() * entropy(y[selected])
    return entropy(y) - remainder


def main() -> None:
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 1, 1])
    print("X shape:", X.shape)
    print("y counts:", np.unique(y, return_counts=True)[1])
    print("parent entropy:", entropy(y))
    for feature_index in range(X.shape[1]):
        print(
            f"feature {feature_index} information gain:",
            information_gain(X[:, feature_index], y),
        )


if __name__ == "__main__":
    main()
