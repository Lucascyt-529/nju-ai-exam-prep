"""逐步展示线性回归的中间量，只调用学生 starter。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])
    w = np.array([0.0])
    b = 0.0
    learning_rate = 0.1

    print("1. X =\n", X, sep="")
    print("   y =", y)
    print("   w =", w, "b =", b)
    print("2. shapes: X", X.shape, "y", y.shape, "w", w.shape)
    print("3. 初始 prediction：调用 starter.predict")
    try:
        prediction = starter.predict(X, w, b)
        print("   实际 prediction =", prediction, "shape =", prediction.shape)
        error = prediction - y
        print("4. error =", error, "shape =", error.shape)
        loss = starter.mean_squared_error(y, prediction)
        print("5. MSE =", loss)
        gradient_w, gradient_b = starter.mse_gradients(X, y, w, b)
        print("6. gradient_w =", gradient_w, "shape =", gradient_w.shape)
        print("   gradient_b =", gradient_b, "type =", type(gradient_b).__name__)
        updated_w = w - learning_rate * gradient_w
        updated_b = b - learning_rate * gradient_b
        print("7. 一次更新后 w =", updated_w, "b =", updated_b)
        updated_loss = starter.mean_squared_error(
            y, starter.predict(X, updated_w, updated_b)
        )
        print("8. 更新后新损失 =", updated_loss)
        gd_w, gd_b, loss_history = starter.fit_gradient_descent(
            X, y, learning_rate=0.05, n_steps=1000
        )
        print("9. 短程梯度下降参数: w =", gd_w, "b =", gd_b)
        print("10. loss_history 前5项 =", loss_history[:5])
        print("    loss_history 后5项 =", loss_history[-5:])
        ls_w, ls_b = starter.fit_least_squares(X, y)
        print("11. 最小二乘结果: w =", ls_w, "b =", ls_b)
        gd_ok = np.allclose(gd_w, [2.0], atol=1e-4) and np.isclose(gd_b, 1.0, atol=1e-4)
        ls_ok = np.allclose(ls_w, [2.0]) and np.isclose(ls_b, 1.0)
        print("12. 期望 w=[2], b=1；梯度下降一致 =", gd_ok, "最小二乘一致 =", ls_ok)
    except NotImplementedError as error:
        print(f"停止展示：{error}")
    except Exception as error:
        print(f"运行到当前步骤时出错：{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
