"""核对 K-means 的最近中心分配。"""

import numpy as np

import starter


def main() -> int:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0], [10.0, 10.0]])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    expected = np.array([0, 0, 1, 1])
    try:
        actual, _ = starter.assign_labels(X, centers)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    correct = np.array_equal(actual, expected)
    print("labels 期望:", expected)
    print("labels 实际:", actual)
    print("一致:", correct)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
