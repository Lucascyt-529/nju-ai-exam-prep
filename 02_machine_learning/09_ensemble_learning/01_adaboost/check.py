"""核对 AdaBoost 三轮训练后的预测。"""

import numpy as np

import starter


def main() -> int:
    X = np.arange(5.0).reshape(-1, 1)
    y = np.array([-1, 1, -1, 1, 1])
    try:
        model = starter.fit_adaboost(X, y, n_estimators=3)
        actual = starter.predict(model, X)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    correct = np.array_equal(actual, y)
    print("prediction 期望:", y)
    print("prediction 实际:", actual)
    print("一致:", correct)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
