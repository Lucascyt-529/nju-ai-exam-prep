"""学生练习：混合数值/类别特征的训练与独立预测程序。"""

from pathlib import Path

import numpy as np


def load_mixed_classification_csv(
    path: Path, *, has_label: bool
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
    raise NotImplementedError("请完成 load_mixed_classification_csv")


def fit_numeric_preprocessor(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 fit_numeric_preprocessor")


def transform_numeric(
    X_numeric: np.ndarray,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError("请完成 transform_numeric")


def fit_vocabularies(X_train: np.ndarray) -> tuple[tuple[str, ...], ...]:
    raise NotImplementedError("请完成 fit_vocabularies")


def transform_one_hot(
    X_categorical: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    raise NotImplementedError("请完成 transform_one_hot")


def build_design_matrix(
    X_numeric: np.ndarray,
    X_categorical: np.ndarray,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    raise NotImplementedError("请完成 build_design_matrix")


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    raise NotImplementedError("请完成 fit_logistic_regression")


def save_encoder_metadata(
    path: Path, vocabularies: tuple[tuple[str, ...], ...]
) -> None:
    raise NotImplementedError("请完成 save_encoder_metadata")


def load_encoder_metadata(path: Path) -> tuple[tuple[str, ...], ...]:
    raise NotImplementedError("请完成 load_encoder_metadata")


def save_model_bundle(
    path: Path,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold: float,
) -> None:
    raise NotImplementedError("请完成 save_model_bundle")


def load_model_bundle(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    raise NotImplementedError("请完成 load_model_bundle")


def run_training(*args, **kwargs) -> None:
    raise NotImplementedError("请完成 run_training")


def run_prediction(*args, **kwargs) -> None:
    raise NotImplementedError("请完成 run_prediction")


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
