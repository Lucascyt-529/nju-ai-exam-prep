"""展示一维 kNN 分类样例。"""

import numpy as np

import starter


def main() -> None:
    X_train = np.array([[0.0], [1.0], [3.0], [4.0]])
    y_train = np.array(["A", "A", "B", "B"])
    X_query = np.array([[0.5], [3.5]])
    expected = np.array(["A", "B"])
    print("查询样本:", X_query.ravel())
    print("期望分类:", expected)
    try:
        actual = starter.predict_classification(X_query, X_train, y_train, 2)
    except NotImplementedError:
        print("实际分类: predict_classification 尚未完成")
        return
    print("实际分类:", actual)
    print("一致:", np.array_equal(actual, expected))


if __name__ == "__main__":
    main()
