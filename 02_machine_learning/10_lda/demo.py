"""展示二分类样本的类别均值。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    expected = np.array([[-1.5, 0.0], [1.5, 1.0]])
    print("期望 class means:")
    print(expected)
    try:
        mean0, mean1 = starter.class_means(X, y)
        actual = np.vstack((mean0, mean1))
    except NotImplementedError:
        print("实际 class means: class_means 尚未完成")
        return
    print("实际 class means:")
    print(actual)
    print("一致:", np.allclose(actual, expected))


if __name__ == "__main__":
    main()
