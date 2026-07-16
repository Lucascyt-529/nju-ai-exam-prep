"""运行前预测特征向量和样本列向量的广播方向。"""

import numpy as np


def main() -> None:
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    feature_offsets = np.array([100.0, 200.0])
    sample_offsets = np.array([1000.0, 2000.0, 3000.0])[:, None]

    print("X shape:", X.shape)
    print("feature offsets shape:", feature_offsets.shape)
    print(X + feature_offsets)
    print("sample offsets shape:", sample_offsets.shape)
    print(X + sample_offsets)

    means = X.mean(axis=0)
    print("means shape:", means.shape)
    print("centered =")
    print(X - means)


if __name__ == "__main__":
    main()
