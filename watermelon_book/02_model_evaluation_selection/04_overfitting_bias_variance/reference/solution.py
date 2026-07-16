"""参考实现：多项式复杂度、学习曲线与偏差—方差。"""

from collections.abc import Sequence

import numpy as np


def _validate_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是非空有限一维数组")
    return array


def _validate_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_array = _validate_vector(x, "x")
    y_array = _validate_vector(y, "y")
    if x_array.shape != y_array.shape:
        raise ValueError("x 和 y 形状必须一致")
    return x_array, y_array


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """返回从 x^0 到 x^degree 的二维设计矩阵。"""
    x_array = _validate_vector(x, "x")
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree 必须是非负整数")
    return np.column_stack([x_array**power for power in range(degree + 1)])


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """返回按升幂排列的最小二乘系数。"""
    x_array, y_array = _validate_xy(x, y)
    design = polynomial_features(x_array, degree)
    coefficients, _, _, _ = np.linalg.lstsq(design, y_array, rcond=None)
    return coefficients


def predict_polynomial(
    x: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    coefficient_array = _validate_vector(coefficients, "coefficients")
    return polynomial_features(x, coefficient_array.size - 1) @ coefficient_array


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _validate_degrees(degrees: Sequence[int]) -> np.ndarray:
    result = np.asarray(list(degrees), dtype=int)
    if result.ndim != 1 or result.size == 0 or np.any(result < 0):
        raise ValueError("degrees 必须是非空非负整数序列")
    if np.unique(result).size != result.size:
        raise ValueError("degrees 不能重复")
    return result


def complexity_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    degrees: Sequence[int],
) -> dict[str, np.ndarray]:
    """返回 degrees、train_mse 和 validation_mse。"""
    train_x, train_y = _validate_xy(x_train, y_train)
    validation_x, validation_y = _validate_xy(x_validation, y_validation)
    degree_array = _validate_degrees(degrees)
    train_mse = np.empty(degree_array.size, dtype=float)
    validation_mse = np.empty(degree_array.size, dtype=float)
    for index, degree in enumerate(degree_array):
        coefficients = fit_polynomial(train_x, train_y, int(degree))
        train_mse[index] = _mse(train_y, predict_polynomial(train_x, coefficients))
        validation_mse[index] = _mse(
            validation_y, predict_polynomial(validation_x, coefficients)
        )
    return {
        "degrees": degree_array,
        "train_mse": train_mse,
        "validation_mse": validation_mse,
    }


def learning_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    degree: int,
    train_sizes: Sequence[int],
) -> dict[str, np.ndarray]:
    train_x, train_y = _validate_xy(x_train, y_train)
    validation_x, validation_y = _validate_xy(x_validation, y_validation)
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree 必须是非负整数")
    sizes = np.asarray(list(train_sizes), dtype=int)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes < degree + 1) or np.any(sizes > train_x.size):
        raise ValueError("train_sizes 必须位于 degree+1 到训练样本数之间")
    if np.any(np.diff(sizes) <= 0):
        raise ValueError("train_sizes 必须严格递增")
    train_mse = np.empty(sizes.size, dtype=float)
    validation_mse = np.empty(sizes.size, dtype=float)
    for index, size in enumerate(sizes):
        coefficients = fit_polynomial(train_x[:size], train_y[:size], degree)
        train_mse[index] = _mse(
            train_y[:size], predict_polynomial(train_x[:size], coefficients)
        )
        validation_mse[index] = _mse(
            validation_y, predict_polynomial(validation_x, coefficients)
        )
    return {
        "train_sizes": sizes,
        "train_mse": train_mse,
        "validation_mse": validation_mse,
    }


def simulate_polynomial_predictions(
    degree: int,
    n_train: int,
    n_repeats: int,
    x_evaluation: np.ndarray,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在 sin(pi*x) 上重复采样，返回预测矩阵与真实值。"""
    evaluation = _validate_vector(x_evaluation, "x_evaluation")
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree 必须是非负整数")
    if not isinstance(n_train, int) or n_train < degree + 1:
        raise ValueError("n_train 至少为 degree+1")
    if not isinstance(n_repeats, int) or n_repeats < 2:
        raise ValueError("n_repeats 至少为2")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std 必须是非负有限数值")
    rng = np.random.default_rng(seed)
    predictions = np.empty((n_repeats, evaluation.size), dtype=float)
    for repeat in range(n_repeats):
        x_train = rng.uniform(-1.0, 1.0, size=n_train)
        y_train = np.sin(np.pi * x_train) + rng.normal(
            0.0, noise_std, size=n_train
        )
        coefficients = fit_polynomial(x_train, y_train, degree)
        predictions[repeat] = predict_polynomial(evaluation, coefficients)
    return predictions, np.sin(np.pi * evaluation)


def bias_variance_components(
    predictions: np.ndarray,
    true_values: np.ndarray,
    noise_variance: float,
) -> dict[str, float | np.ndarray]:
    prediction_array = np.asarray(predictions, dtype=float)
    truth = _validate_vector(true_values, "true_values")
    if prediction_array.ndim != 2 or prediction_array.shape[0] < 2 or prediction_array.shape[1] != truth.size:
        raise ValueError("predictions 必须是 (n_repeats, n_points) 的二维数组")
    if not np.all(np.isfinite(prediction_array)):
        raise ValueError("predictions 必须只包含有限数值")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance 必须是非负有限数值")
    mean_prediction = prediction_array.mean(axis=0)
    pointwise_bias_squared = (mean_prediction - truth) ** 2
    pointwise_variance = prediction_array.var(axis=0)
    bias_squared = float(pointwise_bias_squared.mean())
    variance = float(pointwise_variance.mean())
    return {
        "mean_prediction": mean_prediction,
        "pointwise_bias_squared": pointwise_bias_squared,
        "pointwise_variance": pointwise_variance,
        "bias_squared": bias_squared,
        "variance": variance,
        "noise": float(noise_variance),
        "expected_error": bias_squared + variance + noise_variance,
    }
