"""核对感知机得分、预测和训练结果。"""

import numpy as np

import starter


def main() -> int:
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([-1, -1, 1, 1])
    weights = np.array([1.0, 1.0])
    passed = 0
    try:
        scores = starter.decision_function(X, weights, 0.0)
        prediction = starter.predict_perceptron(X, weights, 0.0)
        fitted_w, fitted_b, _ = starter.train_perceptron(X, y)
        fitted_prediction = starter.predict_perceptron(X, fitted_w, fitted_b)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1
    passed += int(np.allclose(scores, [-3.0, -3.0, 2.0, 3.0]))
    passed += int(np.array_equal(prediction, y))
    passed += int(np.array_equal(fitted_prediction, y))
    print("scores 期望/实际: [-3 -3 2 3] /", scores)
    print("固定参数 prediction:", prediction)
    print("训练后 prediction:", fitted_prediction)
    print(f"通过: {passed}/3")
    return 0 if passed == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
