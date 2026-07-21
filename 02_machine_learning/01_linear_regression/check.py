"""按实现顺序打印期望值和实际值。"""

import numpy as np

import starter


def compare(name: str, actual: object, expected: object) -> bool:
    correct = np.allclose(actual, expected, rtol=1e-5, atol=1e-7)
    print(f"\n{name}")
    print("期望:", expected)
    print("实际:", actual)
    print("一致:", correct)
    return bool(correct)


def main() -> int:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])
    w = np.array([2.0])
    b = 1.0
    passed = 0

    try:
        prediction = starter.predict(X, w, b)
        passed += compare("predict", prediction, y)

        mse = starter.mean_squared_error(y, prediction)
        passed += compare("mean_squared_error", mse, 0.0)

        gradient_w, gradient_b = starter.mse_gradients(X, y, w, b)
        passed += compare("mse_gradients", [gradient_w[0], gradient_b], [0.0, 0.0])

        gd_w, gd_b, _ = starter.fit_gradient_descent(
            X, y, learning_rate=0.03, n_steps=2000
        )
        passed += compare("fit_gradient_descent", [gd_w[0], gd_b], [2.0, 1.0])

        ls_w, ls_b = starter.fit_least_squares(X, y)
        passed += compare("fit_least_squares", [ls_w[0], ls_b], [2.0, 1.0])
    except NotImplementedError as error:
        print(f"\n停止核对：{error}")
    except Exception as error:
        print(f"\n运行错误：{error}")

    print(f"\n通过: {passed}/5")
    return 0 if passed == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
