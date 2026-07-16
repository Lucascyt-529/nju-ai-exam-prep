"""学生练习：线性回归的完整CSV训练、验证、保存与预测程序。"""

from pathlib import Path

import numpy as np


def load_regression_csv(
    path: Path, *, has_target: bool
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    raise NotImplementedError("请完成 load_regression_csv")


def fit_least_squares(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    raise NotImplementedError("请完成 fit_least_squares")


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    raise NotImplementedError("请完成 predict")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    raise NotImplementedError("请完成 regression_metrics")


def save_model(path: Path, weights: np.ndarray, bias: float) -> None:
    raise NotImplementedError("请完成 save_model")


def load_model(path: Path) -> tuple[np.ndarray, float]:
    raise NotImplementedError("请完成 load_model")


def save_predictions(
    path: Path, sample_ids: list[str], predictions: np.ndarray
) -> None:
    raise NotImplementedError("请完成 save_predictions")


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    raise NotImplementedError("请完成 save_metrics")


def run_pipeline(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    model_path: Path,
) -> None:
    raise NotImplementedError("请完成 run_pipeline")


def main() -> int:
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
