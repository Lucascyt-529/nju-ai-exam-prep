"""核对离散朴素贝叶斯的训练和预测。"""

import numpy as np

import starter


def main() -> int:
    X = np.array([["sunny"], ["sunny"], ["rainy"], ["rainy"]])
    y = np.array([0, 0, 1, 1])
    query = np.array([["sunny"], ["rainy"]])
    expected = np.array([0, 1])
    try:
        model = starter.fit_categorical_nb(X, y)
        actual = starter.predict(model, query)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    correct = np.array_equal(actual, expected)
    print("prediction 期望:", expected)
    print("prediction 实际:", actual)
    print("一致:", correct)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
