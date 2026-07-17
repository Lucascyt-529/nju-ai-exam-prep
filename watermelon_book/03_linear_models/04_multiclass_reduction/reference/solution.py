"""参考实现：NumPy实现OvR、OvO与ECOC多分类拆分。"""

import numpy as np


def _validate_training_data(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X必须是非空二维数组")
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y必须是一维且样本数与X一致")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必须只包含有限数值")
    if not np.issubdtype(y.dtype, np.number) or not np.all(np.isfinite(y)):
        raise ValueError("y必须包含有限数值标签")
    classes = np.unique(y)
    if classes.size < 3:
        raise ValueError("多分类专题要求至少三个类别")
    return classes


def _validate_prediction_X(X: np.ndarray, n_features: int) -> None:
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] != n_features
    ):
        raise ValueError("X必须是特征数匹配的非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必须只包含有限数值")


def _validate_classes(classes: np.ndarray) -> None:
    if not isinstance(classes, np.ndarray) or classes.ndim != 1 or classes.size < 3:
        raise ValueError("classes必须是至少包含三个标签的一维数组")
    if not np.issubdtype(classes.dtype, np.number) or not np.all(np.isfinite(classes)):
        raise ValueError("classes必须包含有限数值标签")
    if not np.array_equal(classes, np.unique(classes)):
        raise ValueError("classes必须按升序排列且不能重复")


def _fit_centroid_binary_score(
    X: np.ndarray, binary_y: np.ndarray
) -> tuple[np.ndarray, float]:
    negative = X[binary_y == 0]
    positive = X[binary_y == 1]
    if negative.shape[0] == 0 or positive.shape[0] == 0:
        raise ValueError("二分类训练必须同时包含正类和负类")
    negative_mean = negative.mean(axis=0)
    positive_mean = positive.mean(axis=0)
    weights = positive_mean - negative_mean
    if np.linalg.norm(weights) == 0:
        raise ValueError("正负类中心相同，无法建立线性中心打分器")
    bias = float(-0.5 * (positive_mean @ positive_mean - negative_mean @ negative_mean))
    return weights, bias


def fit_ovr(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回classes:(K,)、weights:(K,d)和biases:(K,)。"""
    classes = _validate_training_data(X, y)
    weights = np.empty((classes.size, X.shape[1]), dtype=float)
    biases = np.empty(classes.size, dtype=float)
    for class_index, label in enumerate(classes):
        binary_y = (y == label).astype(int)
        weights[class_index], biases[class_index] = _fit_centroid_binary_score(X, binary_y)
    return classes, weights, biases


def decision_function_ovr(
    X: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    if not isinstance(weights, np.ndarray) or weights.ndim != 2 or weights.shape[0] < 3:
        raise ValueError("weights必须具有形状(K,d)，且K至少为3")
    if not isinstance(biases, np.ndarray) or biases.shape != (weights.shape[0],):
        raise ValueError("biases必须具有形状(K,)")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(biases)):
        raise ValueError("模型参数必须有限")
    _validate_prediction_X(X, weights.shape[1])
    return X @ weights.T + biases


def predict_ovr(
    X: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    _validate_classes(classes)
    if not isinstance(weights, np.ndarray) or weights.ndim != 2:
        raise ValueError("weights必须具有形状(K,d)")
    if classes.shape[0] != weights.shape[0]:
        raise ValueError("classes必须与分类器数量一致")
    scores = decision_function_ovr(X, weights, biases)
    return classes[np.argmax(scores, axis=1)]


def _expected_class_pairs(n_classes: int) -> np.ndarray:
    return np.array(
        [
            (left, right)
            for left in range(n_classes)
            for right in range(left + 1, n_classes)
        ],
        dtype=int,
    )


def fit_ovo(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回classes、内部类别下标对、weights和biases。"""
    classes = _validate_training_data(X, y)
    class_pairs = _expected_class_pairs(classes.size)
    weights = np.empty((class_pairs.shape[0], X.shape[1]), dtype=float)
    biases = np.empty(class_pairs.shape[0], dtype=float)
    for pair_index, (left, right) in enumerate(class_pairs):
        left_label = classes[left]
        right_label = classes[right]
        selected = (y == left_label) | (y == right_label)
        binary_y = (y[selected] == right_label).astype(int)
        weights[pair_index], biases[pair_index] = _fit_centroid_binary_score(
            X[selected], binary_y
        )
    return classes, class_pairs, weights, biases


def _validate_ovo_model(
    class_pairs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    n_classes: int,
) -> None:
    expected = _expected_class_pairs(n_classes)
    if not isinstance(class_pairs, np.ndarray) or not np.array_equal(class_pairs, expected):
        raise ValueError("class_pairs必须按升序包含全部类别下标对")
    if not isinstance(weights, np.ndarray) or weights.ndim != 2 or weights.shape[0] != len(expected):
        raise ValueError("weights必须为每个类别对提供一个权重向量")
    if not isinstance(biases, np.ndarray) or biases.shape != (len(expected),):
        raise ValueError("biases必须为每个类别对提供一个偏置")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(biases)):
        raise ValueError("模型参数必须有限")


def decision_function_ovo(
    X: np.ndarray,
    class_pairs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    if not isinstance(class_pairs, np.ndarray) or class_pairs.ndim != 2 or class_pairs.shape[1:] != (2,):
        raise ValueError("class_pairs必须具有形状(P,2)")
    n_classes = int(class_pairs.max()) + 1 if class_pairs.size else 0
    if n_classes < 3:
        raise ValueError("OvO模型必须至少包含三个类别")
    _validate_ovo_model(class_pairs, weights, biases, n_classes)
    _validate_prediction_X(X, weights.shape[1])
    return X @ weights.T + biases


def ovo_vote_counts(
    pair_scores: np.ndarray, class_pairs: np.ndarray, n_classes: int
) -> np.ndarray:
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 3:
        raise ValueError("n_classes必须是至少为3的整数")
    expected = _expected_class_pairs(int(n_classes))
    if not isinstance(class_pairs, np.ndarray) or not np.array_equal(class_pairs, expected):
        raise ValueError("class_pairs与n_classes不一致")
    if (
        not isinstance(pair_scores, np.ndarray)
        or pair_scores.ndim != 2
        or pair_scores.shape[0] == 0
        or pair_scores.shape[1] != len(expected)
        or not np.all(np.isfinite(pair_scores))
    ):
        raise ValueError("pair_scores必须是形状(n,P)的非空有限二维数组")

    votes = np.zeros((pair_scores.shape[0], int(n_classes)), dtype=int)
    sample_indices = np.arange(pair_scores.shape[0])
    for pair_index, (left, right) in enumerate(class_pairs):
        winners = np.where(pair_scores[:, pair_index] > 0, right, left)
        votes[sample_indices, winners] += 1
    return votes


def predict_ovo(
    X: np.ndarray,
    classes: np.ndarray,
    class_pairs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    _validate_classes(classes)
    _validate_ovo_model(class_pairs, weights, biases, classes.size)
    scores = decision_function_ovo(X, class_pairs, weights, biases)
    votes = ovo_vote_counts(scores, class_pairs, classes.size)
    return classes[np.argmax(votes, axis=1)]


def _validate_n_classes(n_classes: int) -> int:
    if (
        not isinstance(n_classes, (int, np.integer))
        or isinstance(n_classes, (bool, np.bool_))
        or n_classes < 3
    ):
        raise ValueError("n_classes必须是至少为3的整数")
    return int(n_classes)


def make_ovr_coding_matrix(n_classes: int) -> np.ndarray:
    """返回K乘K二元码：每列一个类别为+1，其余类别为-1。"""
    count = _validate_n_classes(n_classes)
    matrix = -np.ones((count, count), dtype=int)
    np.fill_diagonal(matrix, 1)
    return matrix


def make_ovo_coding_matrix(n_classes: int) -> np.ndarray:
    """返回OvO三元码：类别对为-1/+1，其余类别在该列停用为0。"""
    count = _validate_n_classes(n_classes)
    pairs = _expected_class_pairs(count)
    matrix = np.zeros((count, len(pairs)), dtype=int)
    for column, (left, right) in enumerate(pairs):
        matrix[left, column] = -1
        matrix[right, column] = 1
    return matrix


def _validate_coding_matrix(coding_matrix: np.ndarray, n_classes: int) -> None:
    _validate_n_classes(n_classes)
    if (
        not isinstance(coding_matrix, np.ndarray)
        or not np.issubdtype(coding_matrix.dtype, np.number)
        or coding_matrix.ndim != 2
        or coding_matrix.shape[0] != n_classes
        or coding_matrix.shape[1] == 0
        or not np.all(np.isfinite(coding_matrix))
        or not np.all(np.isin(coding_matrix, [-1, 0, 1]))
    ):
        raise ValueError("coding_matrix必须是形状(K,M)且只含-1、0、+1的数值矩阵")
    for column in range(coding_matrix.shape[1]):
        values = coding_matrix[:, column]
        if not np.any(values == -1) or not np.any(values == 1):
            raise ValueError("编码矩阵每一列都必须同时包含正类+1和反类-1")
    if np.unique(coding_matrix, axis=0).shape[0] != n_classes:
        raise ValueError("每个类别的编码行必须唯一")


def ecoc_training_targets(
    y: np.ndarray,
    classes: np.ndarray,
    coding_matrix: np.ndarray,
    classifier_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回某一ECOC列使用的样本掩码和0/1二分类目标。"""
    _validate_classes(classes)
    _validate_coding_matrix(coding_matrix, classes.size)
    if (
        not isinstance(y, np.ndarray)
        or y.ndim != 1
        or y.size == 0
        or not np.issubdtype(y.dtype, np.number)
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("y必须是非空有限数值标签一维数组")
    if (
        not isinstance(classifier_index, (int, np.integer))
        or isinstance(classifier_index, (bool, np.bool_))
        or not 0 <= classifier_index < coding_matrix.shape[1]
    ):
        raise ValueError("classifier_index超出编码列范围")
    class_indices = np.searchsorted(classes, y)
    if np.any(class_indices == classes.size):
        raise ValueError("y包含classes中不存在的标签")
    if not np.array_equal(classes[class_indices], y):
        raise ValueError("y包含classes中不存在的标签")
    sample_codes = coding_matrix[class_indices, int(classifier_index)]
    selected = sample_codes != 0
    targets = (sample_codes[selected] == 1).astype(int)
    return selected, targets


def fit_ecoc(
    X: np.ndarray, y: np.ndarray, coding_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按给定二元/三元编码训练每一列的中心线性二分类器。"""
    classes = _validate_training_data(X, y)
    _validate_coding_matrix(coding_matrix, classes.size)
    matrix = coding_matrix.astype(int, copy=True)
    weights = np.empty((matrix.shape[1], X.shape[1]), dtype=float)
    biases = np.empty(matrix.shape[1], dtype=float)
    for classifier_index in range(matrix.shape[1]):
        selected, binary_y = ecoc_training_targets(
            y, classes, matrix, classifier_index
        )
        weights[classifier_index], biases[classifier_index] = (
            _fit_centroid_binary_score(X[selected], binary_y)
        )
    return classes, matrix, weights, biases


def _validate_ecoc_model(
    coding_matrix: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    n_classes: int,
) -> None:
    _validate_coding_matrix(coding_matrix, n_classes)
    n_classifiers = coding_matrix.shape[1]
    if (
        not isinstance(weights, np.ndarray)
        or weights.ndim != 2
        or weights.shape[0] != n_classifiers
        or weights.shape[1] == 0
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("weights必须为每个ECOC编码列提供一个有限权重向量")
    if (
        not isinstance(biases, np.ndarray)
        or biases.shape != (n_classifiers,)
        or not np.all(np.isfinite(biases))
    ):
        raise ValueError("biases必须为每个ECOC编码列提供一个有限偏置")


def decision_function_ecoc(
    X: np.ndarray,
    coding_matrix: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    if not isinstance(coding_matrix, np.ndarray) or coding_matrix.ndim != 2:
        raise ValueError("coding_matrix必须是二维数组")
    _validate_ecoc_model(
        coding_matrix, weights, biases, coding_matrix.shape[0]
    )
    _validate_prediction_X(X, weights.shape[1])
    return X @ weights.T + biases


def hard_ecoc_codes(scores: np.ndarray) -> np.ndarray:
    """把二分类分数变为-1/+1；分数等于0时确定地取+1。"""
    if (
        not isinstance(scores, np.ndarray)
        or scores.ndim != 2
        or 0 in scores.shape
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("scores必须是非空有限二维数组")
    return np.where(scores >= 0.0, 1, -1)


def ecoc_distances(
    predicted_codes: np.ndarray,
    coding_matrix: np.ndarray,
    *,
    metric: str = "hamming",
) -> np.ndarray:
    """计算每个预测码到每个类别编码行的海明或欧氏距离。"""
    if (
        not isinstance(predicted_codes, np.ndarray)
        or predicted_codes.ndim != 2
        or 0 in predicted_codes.shape
        or not np.all(np.isin(predicted_codes, [-1, 1]))
    ):
        raise ValueError("predicted_codes必须是只含-1/+1的非空二维数组")
    if not isinstance(coding_matrix, np.ndarray) or coding_matrix.ndim != 2:
        raise ValueError("coding_matrix必须是二维数组")
    _validate_coding_matrix(coding_matrix, coding_matrix.shape[0])
    if predicted_codes.shape[1] != coding_matrix.shape[1]:
        raise ValueError("预测码长度必须与编码矩阵列数一致")
    if metric not in {"hamming", "euclidean"}:
        raise ValueError("metric只能是hamming或euclidean")
    differences = predicted_codes[:, None, :] - coding_matrix[None, :, :]
    if metric == "hamming":
        return np.sum(differences != 0, axis=2).astype(float)
    return np.sqrt(np.sum(differences.astype(float) ** 2, axis=2))


def decode_ecoc(
    predicted_codes: np.ndarray,
    classes: np.ndarray,
    coding_matrix: np.ndarray,
    *,
    metric: str = "hamming",
) -> np.ndarray:
    _validate_classes(classes)
    _validate_coding_matrix(coding_matrix, classes.size)
    distances = ecoc_distances(
        predicted_codes, coding_matrix, metric=metric
    )
    return classes[np.argmin(distances, axis=1)]


def predict_ecoc(
    X: np.ndarray,
    classes: np.ndarray,
    coding_matrix: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    *,
    metric: str = "hamming",
) -> np.ndarray:
    _validate_classes(classes)
    _validate_ecoc_model(
        coding_matrix, weights, biases, classes.size
    )
    scores = decision_function_ecoc(X, coding_matrix, weights, biases)
    return decode_ecoc(
        hard_ecoc_codes(scores), classes, coding_matrix, metric=metric
    )


def minimum_hamming_distance(coding_matrix: np.ndarray) -> int:
    """返回任意两个类别编码行之间的最小海明距离。"""
    if not isinstance(coding_matrix, np.ndarray) or coding_matrix.ndim != 2:
        raise ValueError("coding_matrix必须是二维数组")
    _validate_coding_matrix(coding_matrix, coding_matrix.shape[0])
    distances = [
        int(np.sum(coding_matrix[left] != coding_matrix[right]))
        for left in range(coding_matrix.shape[0])
        for right in range(left + 1, coding_matrix.shape[0])
    ]
    return min(distances)


def binary_error_correction_capacity(coding_matrix: np.ndarray) -> int:
    """二元码在独立位错误理想条件下可保证纠正的错误位数。"""
    if not isinstance(coding_matrix, np.ndarray) or np.any(coding_matrix == 0):
        raise ValueError("纠错位数公式这里只用于不含停用类0的二元码")
    return (minimum_hamming_distance(coding_matrix) - 1) // 2
