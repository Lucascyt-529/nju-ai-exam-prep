"""核对固定随机种子的 Bootstrap 下标。"""

import numpy as np

import starter


def main() -> int:
    expected = np.array([[4, 3, 2, 1, 1], [0, 0, 0, 0, 4]])
    try:
        actual = starter.bootstrap_sample_indices(5, 2, random_state=0)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    correct = np.array_equal(actual, expected)
    print("indices 期望:")
    print(expected)
    print("indices 实际:")
    print(actual)
    print("一致:", correct)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
