"""核对线性核和 SMO 训练预测。"""

import numpy as np

import starter


def main() -> int:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1, -1, 1, 1])
    try:
        gram = starter.linear_kernel_matrix(X, X)
        model = starter.fit_linear_svm_smo(X, y)
        prediction = np.where(starter.decision_function(model, X) >= 0.0, 1, -1)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    gram_ok = np.allclose(gram, X @ X.T)
    prediction_ok = np.array_equal(prediction, y)
    print("Gram matrix 一致:", gram_ok)
    print("prediction 期望:", y)
    print("prediction 实际:", prediction)
    print("一致:", prediction_ok)
    return 0 if gram_ok and prediction_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
