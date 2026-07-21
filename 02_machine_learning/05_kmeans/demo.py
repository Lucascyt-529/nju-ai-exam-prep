"""展示给定两个中心时的最近中心分配。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0], [10.0, 10.0]])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    expected = np.array([0, 0, 1, 1])
    print("期望 labels:", expected)
    try:
        actual, _ = starter.assign_labels(X, centers)
    except NotImplementedError:
        print("实际 labels: assign_labels 尚未完成")
        return
    print("实际 labels:", actual)
    print("一致:", np.array_equal(actual, expected))


if __name__ == "__main__":
    main()
