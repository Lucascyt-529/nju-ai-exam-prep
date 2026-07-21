"""核对固定随机种子的候选特征子集。"""

import numpy as np

import starter


def main() -> int:
    expected = np.array([[2, 3], [0, 1], [0, 3]])
    try:
        actual = starter.sample_feature_subsets(4, 3, 2, random_state=0)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    correct = np.array_equal(actual, expected)
    print("feature subsets 期望:")
    print(expected)
    print("feature subsets 实际:")
    print(actual)
    print("一致:", correct)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
