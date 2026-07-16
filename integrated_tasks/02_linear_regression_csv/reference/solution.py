"""参考实现：线性回归的完整CSV训练、验证、保存与预测程序。"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = ["feature_1", "feature_2"]
TRAIN_HEADER = ["sample_id", *FEATURE_COLUMNS, "target"]
TEST_HEADER = ["sample_id", *FEATURE_COLUMNS]
MODEL_KEYS = {"weights", "bias"}


def load_regression_csv(
    path: Path, *, has_target: bool
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    expected_header = TRAIN_HEADER if has_target else TEST_HEADER
    sample_ids: list[str] = []
    rows: list[list[float]] = []
    targets: list[float] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_header:
            raise ValueError(f"表头必须按以下顺序出现: {expected_header}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第 {line_number} 行字段数量错误")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"第 {line_number} 行 sample_id 为空或重复")
            try:
                features = [float(row[column]) for column in FEATURE_COLUMNS]
                target = float(row["target"]) if has_target else None
            except ValueError as exc:
                raise ValueError(f"第 {line_number} 行包含非法数值") from exc
            values = features + ([] if target is None else [target])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"第 {line_number} 行包含非有限数值")
            sample_ids.append(sample_id)
            rows.append(features)
            if target is not None:
                targets.append(target)
            seen_ids.add(sample_id)

    if not rows:
        raise ValueError("CSV 中没有数据行")
    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float) if has_target else None
    return sample_ids, X, y


def _validate_X(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 必须只包含有限数值")


def fit_least_squares(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    _validate_X(X)
    if not isinstance(y, np.ndarray) or y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y 必须具有形状 (n_samples,)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y 必须只包含有限数值")
    design = np.column_stack((np.ones(X.shape[0]), X))
    parameters, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return parameters[1:].copy(), float(parameters[0])


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    _validate_X(X)
    if weights.ndim != 1 or weights.shape[0] != X.shape[1]:
        raise ValueError("weights 必须具有形状 (n_features,)")
    if not np.all(np.isfinite(weights)) or not np.isfinite(bias):
        raise ValueError("模型参数必须只包含有限数值")
    return X @ weights + bias


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim != 1 or pred.ndim != 1 or true.size == 0 or true.shape != pred.shape:
        raise ValueError("目标和预测必须是形状一致的非空一维数组")
    residual_sum = float(np.sum((true - pred) ** 2))
    total_sum = float(np.sum((true - true.mean()) ** 2))
    mse = residual_sum / true.size
    r2 = (1.0 if residual_sum == 0 else 0.0) if total_sum == 0 else 1.0 - residual_sum / total_sum
    return {"validation_mse": mse, "validation_r2": r2}


def save_model(path: Path, weights: np.ndarray, bias: float) -> None:
    if weights.ndim != 1 or weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("weights 必须是非空有限一维数组")
    if not np.isfinite(bias):
        raise ValueError("bias 必须是有限标量")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, weights=weights, bias=np.asarray(bias, dtype=float))


def load_model(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != MODEL_KEYS:
            raise ValueError(f"模型文件必须包含且只包含: {sorted(MODEL_KEYS)}")
        weights = archive["weights"].astype(float, copy=True)
        bias_array = archive["bias"].astype(float, copy=True)
    if weights.ndim != 1 or weights.size == 0 or bias_array.shape != ():
        raise ValueError("模型参数形状错误")
    bias = float(bias_array)
    if not np.all(np.isfinite(weights)) or not np.isfinite(bias):
        raise ValueError("模型参数包含非有限数值")
    return weights, bias


def save_predictions(
    path: Path, sample_ids: list[str], predictions: np.ndarray
) -> None:
    if predictions.ndim != 1 or predictions.shape[0] != len(sample_ids):
        raise ValueError("预测必须是一维且与 sample_ids 长度一致")
    if len(set(sample_ids)) != len(sample_ids) or not np.all(np.isfinite(predictions)):
        raise ValueError("sample_ids 必须唯一且预测必须有限")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", "prediction"])
        for sample_id, value in zip(sample_ids, predictions, strict=True):
            writer.writerow([sample_id, f"{value:.6f}"])


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    expected = ["validation_mse", "validation_r2"]
    if list(metrics) != expected or not all(np.isfinite(metrics[key]) for key in expected):
        raise ValueError(f"metrics 必须按顺序包含: {expected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{key}={metrics[key]:.6f}\n" for key in expected)
    path.write_text(content, encoding="utf-8", newline="\n")


def run_pipeline(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    model_path: Path,
) -> None:
    _, X_train, y_train = load_regression_csv(train_path, has_target=True)
    _, X_validation, y_validation = load_regression_csv(validation_path, has_target=True)
    test_ids, X_test, _ = load_regression_csv(test_path, has_target=False)
    if y_train is None or y_validation is None:
        raise RuntimeError("训练或验证目标缺失")
    weights, bias = fit_least_squares(X_train, y_train)
    validation_predictions = predict(X_validation, weights, bias)
    metrics = regression_metrics(y_validation, validation_predictions)
    save_model(model_path, weights, bias)
    loaded_weights, loaded_bias = load_model(model_path)
    test_predictions = predict(X_test, loaded_weights, loaded_bias)
    save_predictions(predictions_path, test_ids, test_predictions)
    save_metrics(metrics_path, metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_pipeline(
            args.train, args.validation, args.test,
            args.predictions, args.metrics, args.model,
        )
    except (OSError, ValueError) as exc:
        print(f"线性回归任务失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
