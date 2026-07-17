"""参考实现：NumPy二分类与多分类LDA。"""

import numpy as np


def _validate_data(X: np.ndarray, y: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X必须是非空二维数组")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y必须是一维且样本数与X一致")
    if not np.all(np.isfinite(X)) or not np.all((y == 0) | (y == 1)):
        raise ValueError("X必须有限且y只能包含0/1")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("训练数据必须同时包含0类和1类")


def _validate_weights(X: np.ndarray, weights: np.ndarray) -> None:
    if not isinstance(weights, np.ndarray) or weights.ndim != 1 or weights.shape[0] != X.shape[1]:
        raise ValueError("weights必须具有形状(n_features,)")
    if not np.all(np.isfinite(weights)) or np.linalg.norm(weights) == 0:
        raise ValueError("weights必须是非零有限向量")


def class_means(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_data(X, y)
    return X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)


def within_class_scatter(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    _validate_data(X, y)
    mean0, mean1 = class_means(X, y)
    centered0 = X[y == 0] - mean0
    centered1 = X[y == 1] - mean1
    return centered0.T @ centered0 + centered1.T @ centered1


def fit_binary_lda(
    X: np.ndarray, y: np.ndarray, *, regularization: float = 0.0
) -> tuple[np.ndarray, float]:
    """返回投影权重和投影中点阈值。"""
    _validate_data(X, y)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization必须是非负有限数值")
    mean0, mean1 = class_means(X, y)
    difference = mean1 - mean0
    if np.linalg.norm(difference) == 0:
        raise ValueError("两类均值相同，无法得到LDA均值差方向")
    scatter = within_class_scatter(X, y)
    adjusted = scatter + regularization * np.eye(X.shape[1])
    weights = np.linalg.pinv(adjusted) @ difference
    if not np.all(np.isfinite(weights)) or np.linalg.norm(weights) == 0:
        raise ValueError("无法得到有效LDA投影方向")
    threshold = float(0.5 * (mean0 @ weights + mean1 @ weights))
    return weights, threshold


def project(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必须只包含有限数值")
    _validate_weights(X, weights)
    return X @ weights


def predict_lda(
    X: np.ndarray, weights: np.ndarray, threshold: float
) -> np.ndarray:
    if not np.isfinite(threshold):
        raise ValueError("threshold必须是有限标量")
    return (project(X, weights) >= threshold).astype(int)


def fisher_ratio(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    _validate_data(X, y)
    _validate_weights(X, weights)
    projections = project(X, weights)
    projected0 = projections[y == 0]
    projected1 = projections[y == 1]
    numerator = float((projected1.mean() - projected0.mean()) ** 2)
    denominator = float(
        np.sum((projected0 - projected0.mean()) ** 2)
        + np.sum((projected1 - projected1.mean()) ** 2)
    )
    if denominator == 0:
        return np.inf if numerator > 0 else 0.0
    return numerator / denominator


def _validate_multiclass_data(
    X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    if (
        not isinstance(X, np.ndarray)
        or not np.issubdtype(X.dtype, np.number)
        or np.issubdtype(X.dtype, np.bool_)
        or X.ndim != 2
        or 0 in X.shape
        or not np.all(np.isfinite(X))
    ):
        raise ValueError("X必须是非空有限数值二维数组")
    if (
        not isinstance(y, np.ndarray)
        or not np.issubdtype(y.dtype, np.number)
        or np.issubdtype(y.dtype, np.bool_)
        or y.shape != (X.shape[0],)
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("y必须是与X样本数一致的有限数值一维标签")
    classes = np.unique(y)
    if classes.size < 3:
        raise ValueError("多分类LDA要求至少三个类别")
    return classes


def multiclass_scatter_matrices(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回classes、全局均值、类均值、Sw、Sb和St。"""
    classes = _validate_multiclass_data(X, y)
    X_float = X.astype(float, copy=False)
    global_mean = X_float.mean(axis=0)
    means = np.empty((classes.size, X.shape[1]), dtype=float)
    within = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    between = np.zeros_like(within)
    for class_index, label in enumerate(classes):
        samples = X_float[y == label]
        mean = samples.mean(axis=0)
        means[class_index] = mean
        centered = samples - mean
        within += centered.T @ centered
        mean_difference = mean - global_mean
        between += samples.shape[0] * np.outer(mean_difference, mean_difference)
    total_centered = X_float - global_mean
    total = total_centered.T @ total_centered
    return classes, global_mean, means, within, between, total


def _fix_projection_signs(projection: np.ndarray) -> np.ndarray:
    result = projection.copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def fit_multiclass_lda(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int | None = None,
    regularization: float = 1e-8,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回classes、投影W、投影类中心和降序广义特征值。"""
    classes, _, means, within, between, _ = multiclass_scatter_matrices(X, y)
    if (
        not isinstance(regularization, (int, float, np.integer, np.floating))
        or isinstance(regularization, (bool, np.bool_))
        or not np.isfinite(regularization)
        or regularization < 0
    ):
        raise ValueError("regularization必须是非负有限数值")
    if (
        not isinstance(tolerance, (int, float, np.integer, np.floating))
        or isinstance(tolerance, (bool, np.bool_))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance必须是正有限数值")
    maximum = min(classes.size - 1, X.shape[1])
    if n_components is not None and (
        not isinstance(n_components, (int, np.integer))
        or isinstance(n_components, (bool, np.bool_))
        or not 1 <= n_components <= maximum
    ):
        raise ValueError(f"n_components必须是1到{maximum}之间的整数")

    adjusted_within = within + float(regularization) * np.eye(X.shape[1])
    within_values, within_vectors = np.linalg.eigh(adjusted_within)
    scale = max(1.0, float(np.max(np.abs(within_values))))
    positive = within_values > float(tolerance) * scale
    if not np.any(positive):
        raise ValueError("类内散度没有可用正方向，请增加regularization")
    whitening = within_vectors[:, positive] / np.sqrt(within_values[positive])
    whitened_between = whitening.T @ between @ whitening
    whitened_between = 0.5 * (whitened_between + whitened_between.T)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened_between)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigen_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    informative = eigenvalues > float(tolerance) * eigen_scale
    available = min(maximum, int(np.count_nonzero(informative)))
    if available == 0:
        raise ValueError("类别均值没有形成有效类间判别方向")
    if n_components is None:
        component_count = available
    elif int(n_components) > available:
        raise ValueError(
            f"数据只有{available}个有效判别方向，不能请求{n_components}维"
        )
    else:
        component_count = int(n_components)

    projection = whitening @ eigenvectors[:, :component_count]
    norms = np.linalg.norm(projection, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("无法得到有限非零多分类LDA投影")
    projection = _fix_projection_signs(projection / norms)
    projected_centroids = means @ projection
    return (
        classes.copy(),
        projection,
        projected_centroids,
        eigenvalues[:component_count].copy(),
    )


def project_multiclass(X: np.ndarray, projection: np.ndarray) -> np.ndarray:
    if (
        not isinstance(X, np.ndarray)
        or not np.issubdtype(X.dtype, np.number)
        or X.ndim != 2
        or 0 in X.shape
        or not np.all(np.isfinite(X))
    ):
        raise ValueError("X必须是非空有限数值二维数组")
    if (
        not isinstance(projection, np.ndarray)
        or not np.issubdtype(projection.dtype, np.number)
        or np.issubdtype(projection.dtype, np.bool_)
        or projection.ndim != 2
        or projection.shape[0] != X.shape[1]
        or projection.shape[1] == 0
        or not np.all(np.isfinite(projection))
    ):
        raise ValueError("projection必须是形状(d,r)的非空有限二维数组")
    return X @ projection


def predict_multiclass_lda(
    X: np.ndarray,
    classes: np.ndarray,
    projection: np.ndarray,
    projected_centroids: np.ndarray,
) -> np.ndarray:
    projected = project_multiclass(X, projection)
    if (
        not isinstance(classes, np.ndarray)
        or classes.ndim != 1
        or classes.size < 3
        or not np.issubdtype(classes.dtype, np.number)
        or not np.all(np.isfinite(classes))
        or not np.array_equal(classes, np.unique(classes))
    ):
        raise ValueError("classes必须是至少三个升序唯一有限数值标签")
    if (
        not isinstance(projected_centroids, np.ndarray)
        or not np.issubdtype(projected_centroids.dtype, np.number)
        or projected_centroids.shape != (classes.size, projection.shape[1])
        or not np.all(np.isfinite(projected_centroids))
    ):
        raise ValueError("projected_centroids形状必须为(K,r)且数值有限")
    squared_distances = np.sum(
        (projected[:, None, :] - projected_centroids[None, :, :]) ** 2,
        axis=2,
    )
    return classes[np.argmin(squared_distances, axis=1)]


def multiclass_trace_ratio(
    projection: np.ndarray,
    within: np.ndarray,
    between: np.ndarray,
) -> float:
    if (
        not isinstance(projection, np.ndarray)
        or projection.ndim != 2
        or 0 in projection.shape
        or not np.all(np.isfinite(projection))
    ):
        raise ValueError("projection必须是非空有限二维数组")
    dimension = projection.shape[0]
    if any(
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (dimension, dimension)
        or not np.all(np.isfinite(matrix))
        for matrix in (within, between)
    ):
        raise ValueError("within和between必须是与投影输入维数一致的有限方阵")
    numerator = float(np.trace(projection.T @ between @ projection))
    denominator = float(np.trace(projection.T @ within @ projection))
    if denominator < 0 or numerator < -1e-12:
        raise ValueError("散度矩阵产生了非法负迹")
    if denominator == 0:
        return np.inf if numerator > 0 else 0.0
    return numerator / denominator
