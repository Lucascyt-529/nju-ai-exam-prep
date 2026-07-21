"""核对二分类 LDA 的类别均值和训练预测。"""

import numpy as np

import starter


def main() -> int:
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    try:
        mean0, mean1 = starter.class_means(X, y)
        weights, threshold = starter.fit_binary_lda(X, y, regularization=1e-8)
        prediction = starter.predict_lda(X, weights, threshold)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    means_ok = np.allclose(np.vstack((mean0, mean1)), [[-1.5, 0.0], [1.5, 1.0]])
    prediction_ok = np.array_equal(prediction, y)
    print("类别均值一致:", means_ok)
    print("prediction 期望:", y)
    print("prediction 实际:", prediction)
    print("一致:", prediction_ok)
    return 0 if means_ok and prediction_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
