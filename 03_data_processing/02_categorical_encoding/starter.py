"""学生练习：训练词表、未知类别、one-hot编码与元数据保存。"""

from pathlib import Path

import numpy as np


def fit_category_vocabularies(X_train: np.ndarray) -> tuple[tuple[str, ...], ...]:
    raise NotImplementedError("请完成 fit_category_vocabularies")


def transform_ordinal(
    X: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    handle_unknown: str = "error",
    unknown_value: int = -1,
) -> np.ndarray:
    raise NotImplementedError("请完成 transform_ordinal")


def transform_one_hot(
    X: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    include_unknown: bool = True,
) -> np.ndarray:
    raise NotImplementedError("请完成 transform_one_hot")


def one_hot_feature_names(
    feature_names: tuple[str, ...],
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    include_unknown: bool = True,
) -> tuple[str, ...]:
    raise NotImplementedError("请完成 one_hot_feature_names")


def combine_numeric_and_categorical(X_numeric: np.ndarray,
                                    X_one_hot: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 combine_numeric_and_categorical")


def save_encoder_metadata(
    path: Path,
    feature_names: tuple[str, ...],
    vocabularies: tuple[tuple[str, ...], ...],
    *,
    include_unknown: bool = True,
) -> None:
    raise NotImplementedError("请完成 save_encoder_metadata")


def load_encoder_metadata(path: Path) -> dict[str, object]:
    raise NotImplementedError("请完成 load_encoder_metadata")
