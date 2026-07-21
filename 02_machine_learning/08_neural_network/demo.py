"""展示感知机线性得分和类别预测。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    weights = np.array([1.0, 1.0])
    expected = np.array([-1, -1, 1, 1])
    print("期望 prediction:", expected)
    try:
        actual = starter.predict_perceptron(X, weights, 0.0)
    except NotImplementedError:
        print("实际 prediction: predict_perceptron 尚未完成")
        return
    print("实际 prediction:", actual)
    print("一致:", np.array_equal(actual, expected))


if __name__ == "__main__":
    main()
