"""展示固定随机种子的候选特征子集。"""

import numpy as np

import starter


def main() -> None:
    expected = np.array([[2, 3], [0, 1], [0, 3]])
    print("期望 feature subsets:")
    print(expected)
    try:
        actual = starter.sample_feature_subsets(4, 3, 2, random_state=0)
    except NotImplementedError:
        print("实际结果: sample_feature_subsets 尚未完成")
        return
    print("实际 feature subsets:")
    print(actual)
    print("一致:", np.array_equal(actual, expected))


if __name__ == "__main__":
    main()
