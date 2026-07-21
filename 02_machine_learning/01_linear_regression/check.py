"""线性回归日常核对：逐项打印期望、实际和通过状态。"""

import numpy as np

import starter


TOTAL = 12


def compare(name: str, actual: object, expected: object) -> bool:
    try:
        passed = bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-6))
    except (TypeError, ValueError):
        passed = actual == expected
    print(f"\n{name}\n期望: {expected}\n实际: {actual}\n通过: {passed}")
    return passed


def main() -> int:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])
    X_before, y_before = X.copy(), y.copy()
    passed = 0
    try:
        prediction = starter.predict(X, np.array([2.0]), 1.0)
        passed += compare("1. predict 数值", prediction, y)
        passed += compare("2. predict shape", prediction.shape, (3,))
        passed += compare("3. MSE", starter.mean_squared_error(y, prediction), 0.0)

        gradient_w, gradient_b = starter.mse_gradients(X, y, np.zeros(1), 0.0)
        passed += compare("4. gradient_w 数值", gradient_w, np.array([-26.0 / 3.0]))
        shape_and_type = gradient_w.shape == (1,) and isinstance(gradient_b, float)
        passed += compare("5. 梯度 shape 与 gradient_b 类型", shape_and_type, True)

        new_w = np.zeros(1) - 0.1 * gradient_w
        new_b = 0.0 - 0.1 * gradient_b
        passed += compare("6. 一次梯度更新", [new_w[0], new_b], [13.0 / 15.0, 0.6])

        gd_w, gd_b, history = starter.fit_gradient_descent(
            X, y, learning_rate=0.05, n_steps=1000
        )
        passed += compare("7. 梯度下降最终参数", [gd_w[0], gd_b], [2.0, 1.0])
        passed += compare("8. loss_history 长度", len(history), 1001)
        passed += compare("9. 损失总体下降", bool(history[-1] < history[0]), True)

        ls_w, ls_b = starter.fit_least_squares(X, y)
        passed += compare("10. 最小二乘", [ls_w[0], ls_b], [2.0, 1.0])

        X2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        y2 = 2.0 * X2[:, 0] - X2[:, 1] + 3.0
        w2, b2 = starter.fit_least_squares(X2, y2)
        passed += compare("11. 双特征样例", [*w2, b2], [2.0, -1.0, 3.0])

        unchanged = np.array_equal(X, X_before) and np.array_equal(y, y_before)
        passed += compare("12. X、y 未被原地修改", unchanged, True)
    except NotImplementedError as error:
        print(f"\n停止核对：{error}")
    except Exception as error:
        print(f"\n运行错误：{type(error).__name__}: {error}")

    print(f"\n通过: {passed}/{TOTAL}")
    return 0 if passed == TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
