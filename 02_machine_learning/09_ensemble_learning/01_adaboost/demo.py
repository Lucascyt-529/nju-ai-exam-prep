"""展示 AdaBoost 增加弱分类器前后的预测。"""

import numpy as np

import starter


def main() -> None:
    X = np.arange(5.0).reshape(-1, 1)
    y = np.array([-1, 1, -1, 1, 1])
    print("期望 3 个弱分类器后的 prediction:", y)
    try:
        model = starter.fit_adaboost(X, y, n_estimators=3)
        actual = starter.predict(model, X)
    except NotImplementedError as error:
        print("实际 prediction:", error)
        return
    print("实际 prediction:", actual)
    print("一致:", np.array_equal(actual, y))


if __name__ == "__main__":
    main()
