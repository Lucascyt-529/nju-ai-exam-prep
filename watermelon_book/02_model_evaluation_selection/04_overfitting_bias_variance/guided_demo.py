"""生成可复现的复杂度曲线和偏差—方差数值。"""

import numpy as np


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([x**power for power in range(degree + 1)])


def fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, degree: int
) -> np.ndarray:
    coefficients, _, _, _ = np.linalg.lstsq(
        polynomial_features(x_train, degree), y_train, rcond=None
    )
    return polynomial_features(x_eval, degree) @ coefficients


def main() -> None:
    rng = np.random.default_rng(42)
    x_train = np.linspace(-1.0, 1.0, 10)
    y_train = np.sin(np.pi * x_train) + rng.normal(0.0, 0.15, size=x_train.size)
    x_validation = np.linspace(-0.95, 0.95, 100)
    y_validation = np.sin(np.pi * x_validation)

    print("degree train_mse validation_mse")
    for degree in range(10):
        train_pred = fit_predict(x_train, y_train, x_train, degree)
        validation_pred = fit_predict(x_train, y_train, x_validation, degree)
        train_mse = np.mean((train_pred - y_train) ** 2)
        validation_mse = np.mean((validation_pred - y_validation) ** 2)
        print(degree, f"{train_mse:.6f}", f"{validation_mse:.6f}")


if __name__ == "__main__":
    main()
