"""展示固定随机种子的 Bootstrap 样本下标。"""

import numpy as np

import starter


def main() -> None:
    expected = np.array([[4, 3, 2, 1, 1], [0, 0, 0, 0, 4]])
    print("期望 Bootstrap indices:")
    print(expected)
    try:
        actual = starter.bootstrap_sample_indices(5, 2, random_state=0)
    except NotImplementedError:
        print("实际结果: bootstrap_sample_indices 尚未完成")
        return
    print("实际 Bootstrap indices:")
    print(actual)
    print("一致:", np.array_equal(actual, expected))


if __name__ == "__main__":
    main()
