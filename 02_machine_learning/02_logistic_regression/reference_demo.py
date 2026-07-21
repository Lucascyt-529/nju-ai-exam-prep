"""预测稳定sigmoid、梯度/Hessian形状并比较两种优化。"""

import numpy as np

from reference.solution import (
    fit_gradient_descent,
    fit_newton,
    logistic_hessian,
    stable_sigmoid,
)


def main() -> None:
    logits = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    print("logits:", logits)
    print("probabilities:", stable_sigmoid(logits))

    X = np.array([[1.0, 2.0], [2.0, -1.0], [0.0, 1.0]])
    y = np.array([1.0, 0.0, 1.0])
    w = np.array([0.2, -0.3])
    b = 0.1
    scores = X @ w + b
    errors = stable_sigmoid(scores) - y
    print("score shape:", scores.shape)
    print("error shape:", errors.shape)
    print("gradient_w shape:", (X.T @ errors).shape)
    hessian = logistic_hessian(X, y, w, b, l2=0.1)
    print("Hessian shape:", hessian.shape)
    print("Hessian symmetric:", np.allclose(hessian, hessian.T))

    X_train = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    y_train = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    _, _, gd_losses = fit_gradient_descent(
        X_train, y_train, learning_rate=0.2, n_steps=100, l2=0.01
    )
    _, _, newton_losses = fit_newton(
        X_train, y_train, n_steps=6, l2=0.01, damping=1e-8
    )
    print("gradient descent: steps/final loss:", 100, round(gd_losses[-1], 6))
    print("Newton: steps/final loss:", 6, round(newton_losses[-1], 6))
    print("Newton non-increasing:", np.all(np.diff(newton_losses) <= 1e-15))


if __name__ == "__main__":
    main()
