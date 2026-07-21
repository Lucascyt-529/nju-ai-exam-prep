"""展示离散朴素贝叶斯分类样例。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([["sunny"], ["sunny"], ["rainy"], ["rainy"]])
    y = np.array([0, 0, 1, 1])
    query = np.array([["sunny"], ["rainy"]])
    expected = np.array([0, 1])
    print("query:", query.ravel())
    print("期望 prediction:", expected)
    try:
        model = starter.fit_categorical_nb(X, y)
        actual = starter.predict(model, query)
    except NotImplementedError as error:
        print("实际 prediction:", error)
        return
    print("实际 prediction:", actual)
    print("一致:", np.array_equal(actual, expected))


if __name__ == "__main__":
    main()
