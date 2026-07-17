"""比较恒等联系与对数联系下的预测形状和输出尺度。"""

import numpy as np

from reference.solution import (
    apply_link,
    fit_log_linear,
    inverse_link,
    linear_predictor,
    mean_squared_link_error,
    predict_log_linear,
)


def main() -> None:
    X = np.arange(5.0).reshape(-1, 1)
    y = np.exp(0.7 * X[:, 0] + 0.2)
    weights, bias = fit_log_linear(X, y)
    eta = linear_predictor(X, weights, bias)
    prediction = predict_log_linear(X, weights, bias)

    print("X / y shapes:", X.shape, y.shape)
    print("fitted weights / bias:", np.round(weights, 6), round(bias, 6))
    print("eta shape:", eta.shape)
    print("eta:", np.round(eta, 6))
    print("positive predictions:", np.round(prediction, 6))
    print("log-scale MSE:", mean_squared_link_error(y, prediction, link="log"))

    probabilities = np.array([0.1, 0.5, 0.9])
    logits = apply_link(probabilities, link="logit")
    print("logit round trip:", np.round(inverse_link(logits, link="logit"), 6))


if __name__ == "__main__":
    main()
