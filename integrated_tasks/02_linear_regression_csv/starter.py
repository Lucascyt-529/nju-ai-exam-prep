"""学生练习：线性回归的完整CSV训练、验证、保存与预测程序。"""

from pathlib import Path

import numpy as np
import csv


def load_regression_csv(
    path: Path, *, has_target: bool
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    sample_ids = []
    features = []
    targets = []

    with path.open('r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            sample_ids.append(row[0])

            if has_target == True:
                features.append(row[1:-1])
                targets.append(row[-1])
            else:
                features.append(row[1:])
                targets.append(None)

    X = np.asarray(features, dtype = float)
    if has_target:
        y = np.asarray(targets, dtype=float)
    else:
        y = None

    return sample_ids, X, y


def fit_least_squares(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    raise NotImplementedError("请完成 fit_least_squares")


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return X @ weights + bias


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    squared_error = error ** 2
    mse = squared_error.mean()
    true_mean = y_true.mean()
    error_true = y_true - true_mean
    bmse = (error_true ** 2).mean()

    if bmse == 0:
        if mse == 0:
            r2 = 1
        else:
            r2 = 0

    else:
        r2 = 1 - mse/bmse

    return {
        "mse": float(mse),
        "r2": float(r2)
    }


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
