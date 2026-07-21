"""展示一个能完美划分类别的离散特征。"""

import numpy as np

import starter


def main() -> None:
    feature = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 1, 1])
    print("feature:", feature)
    print("y:", y)
    print("期望 information_gain: 1.0")
    try:
        actual = starter.information_gain(feature, y)
    except NotImplementedError:
        print("实际 information_gain: 尚未完成")
        return
    print("实际 information_gain:", actual)
    print("一致:", np.isclose(actual, 1.0))


if __name__ == "__main__":
    main()
