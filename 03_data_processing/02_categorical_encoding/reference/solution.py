"""参考实现：只用训练集拟合类别词表，并进行编码与安全持久化。"""

import json
from pathlib import Path

import numpy as np


def _categorical_matrix(X: np.ndarray, name: str) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape:
        raise ValueError(f"{name}必须是非空二维数组")
    for value in X.ravel():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}中的类别必须是非空字符串")


def fit_category_vocabularies(X_train: np.ndarray) -> tuple[tuple[str, ...], ...]:
    """逐特征返回排序后的训练类别；验证/测试集不得参与。"""
    _categorical_matrix(X_train, "X_train")
    return tuple(tuple(sorted(set(X_train[:, column].tolist())))
                 for column in range(X_train.shape[1]))


def _validate_vocabularies(vocabularies: tuple[tuple[str, ...], ...],
                           n_features: int) -> None:
    if not isinstance(vocabularies, tuple) or len(vocabularies) != n_features:
        raise ValueError("vocabularies数量必须与类别特征数一致")
    for vocabulary in vocabularies:
        if (not isinstance(vocabulary, tuple) or not vocabulary
                or any(not isinstance(value, str) or not value for value in vocabulary)
                or tuple(sorted(set(vocabulary))) != vocabulary):
            raise ValueError("每个词表必须是非空、唯一、升序字符串元组")


def transform_ordinal(
    X: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    handle_unknown: str = "error",
    unknown_value: int = -1,
) -> np.ndarray:
    """按训练词表编码；未知类别可报错或映射到指定整数。"""
    _categorical_matrix(X, "X")
    _validate_vocabularies(vocabularies, X.shape[1])
    if handle_unknown not in {"error", "use_encoded_value"}:
        raise ValueError("handle_unknown只能为error或use_encoded_value")
    if isinstance(unknown_value, (bool, np.bool_)) or not isinstance(unknown_value, (int, np.integer)):
        raise ValueError("unknown_value必须是整数")
    mappings = [{value: index for index, value in enumerate(vocabulary)}
                for vocabulary in vocabularies]
    encoded = np.empty(X.shape, dtype=int)
    for row in range(X.shape[0]):
        for column in range(X.shape[1]):
            value = X[row, column]
            if value in mappings[column]:
                encoded[row, column] = mappings[column][value]
            elif handle_unknown == "use_encoded_value":
                encoded[row, column] = int(unknown_value)
            else:
                raise ValueError(f"第{column}列出现训练时未见类别: {value}")
    return encoded


def one_hot_feature_names(
    feature_names: tuple[str, ...],
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    include_unknown: bool = True,
) -> tuple[str, ...]:
    if (not isinstance(feature_names, tuple) or len(feature_names) != len(vocabularies)
            or any(not isinstance(name, str) or not name for name in feature_names)
            or len(set(feature_names)) != len(feature_names)):
        raise ValueError("feature_names必须是与词表数一致的唯一非空字符串元组")
    _validate_vocabularies(vocabularies, len(feature_names))
    if not isinstance(include_unknown, (bool, np.bool_)):
        raise ValueError("include_unknown必须是布尔值")
    names: list[str] = []
    for feature, vocabulary in zip(feature_names, vocabularies, strict=True):
        names.extend(f"{feature}={category}" for category in vocabulary)
        if include_unknown:
            names.append(f"{feature}=<UNKNOWN>")
    return tuple(names)


def transform_one_hot(
    X: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    include_unknown: bool = True,
) -> np.ndarray:
    """每个类别特征形成一个块；未知桶位于该块最后一列。"""
    _categorical_matrix(X, "X")
    _validate_vocabularies(vocabularies, X.shape[1])
    if not isinstance(include_unknown, (bool, np.bool_)):
        raise ValueError("include_unknown必须是布尔值")
    blocks: list[np.ndarray] = []
    for column, vocabulary in enumerate(vocabularies):
        mapping = {value: index for index, value in enumerate(vocabulary)}
        width = len(vocabulary) + int(include_unknown)
        block = np.zeros((X.shape[0], width), dtype=float)
        for row, value in enumerate(X[:, column]):
            if value in mapping:
                block[row, mapping[value]] = 1.0
            elif include_unknown:
                block[row, -1] = 1.0
            else:
                raise ValueError(f"第{column}列出现训练时未见类别: {value}")
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def combine_numeric_and_categorical(X_numeric: np.ndarray,
                                    X_one_hot: np.ndarray) -> np.ndarray:
    if (not isinstance(X_numeric, np.ndarray) or X_numeric.ndim != 2
            or not isinstance(X_one_hot, np.ndarray) or X_one_hot.ndim != 2
            or X_numeric.shape[0] != X_one_hot.shape[0] or X_numeric.shape[0] == 0
            or not np.issubdtype(X_numeric.dtype, np.number)
            or not np.issubdtype(X_one_hot.dtype, np.number)
            or not np.all(np.isfinite(X_numeric)) or not np.all(np.isfinite(X_one_hot))):
        raise ValueError("数值特征与one-hot必须是样本数一致的非空有限二维数组")
    return np.concatenate((X_numeric.astype(float), X_one_hot.astype(float)), axis=1)


def save_encoder_metadata(
    path: Path,
    feature_names: tuple[str, ...],
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    include_unknown: bool = True,
) -> None:
    """用UTF-8 JSON保存列语义；不使用pickle。"""
    one_hot_feature_names(
        feature_names, vocabularies, include_unknown=include_unknown
    )
    payload = {
        "format_version": 1,
        "feature_names": list(feature_names),
        "vocabularies": [list(vocabulary) for vocabulary in vocabularies],
        "include_unknown": bool(include_unknown),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def load_encoder_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("编码器元数据不是合法JSON") from exc
    expected_keys = {"format_version", "feature_names", "vocabularies", "include_unknown"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("编码器元数据键集合错误")
    if payload["format_version"] != 1 or isinstance(payload["format_version"], bool):
        raise ValueError("不支持的编码器格式版本")
    if not isinstance(payload["feature_names"], list) or not isinstance(payload["vocabularies"], list):
        raise ValueError("feature_names和vocabularies必须是JSON数组")
    if not isinstance(payload["include_unknown"], bool):
        raise ValueError("include_unknown必须是布尔值")
    try:
        feature_names = tuple(payload["feature_names"])
        vocabularies = tuple(tuple(values) for values in payload["vocabularies"])
    except TypeError as exc:
        raise ValueError("vocabularies必须是二维字符串数组") from exc
    one_hot_feature_names(
        feature_names, vocabularies,
        include_unknown=payload["include_unknown"],
    )
    return {
        "feature_names": feature_names,
        "vocabularies": vocabularies,
        "include_unknown": payload["include_unknown"],
    }
