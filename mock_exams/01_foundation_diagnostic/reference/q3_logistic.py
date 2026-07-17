"""模拟卷1第3题参考实现。"""

import argparse
import csv
from numbers import Real
from pathlib import Path
import sys

import numpy as np


FEATURES = ["feature_1", "feature_2", "feature_3"]
TRAIN_HEADER = ["sample_id", *FEATURES, "label"]
TEST_HEADER = ["sample_id", *FEATURES]
MODEL_KEYS = {"mean", "std", "weights", "bias"}


def load_csv(path: Path, *, has_label: bool) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    expected = TRAIN_HEADER if has_label else TEST_HEADER
    ids: list[str] = []; rows: list[list[float]] = []; labels: list[int] = []; seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected:
            raise ValueError(f"表头必须为{expected}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第{line_number}行字段数量错误")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"第{line_number}行ID为空或重复")
            try:
                features = [float(row[name]) for name in FEATURES]
            except ValueError as exc:
                raise ValueError(f"第{line_number}行特征非法") from exc
            if not np.all(np.isfinite(features)):
                raise ValueError(f"第{line_number}行特征必须有限")
            if has_label:
                try:
                    label = int(row["label"])
                except ValueError as exc:
                    raise ValueError(f"第{line_number}行标签非法") from exc
                if row["label"].strip() not in {"0", "1"}:
                    raise ValueError(f"第{line_number}行标签只能为0或1")
                labels.append(label)
            ids.append(sample_id); rows.append(features); seen.add(sample_id)
    if not rows:
        raise ValueError("CSV不能为空")
    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=float) if has_label else None
    return ids, X, y


def fit_preprocessor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape or not np.all(np.isfinite(X)):
        raise ValueError("X必须是非空有限二维数组")
    mean = np.mean(X, axis=0); std = np.std(X, axis=0, ddof=0)
    return mean, np.where(std == 0.0, 1.0, std)


def transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or mean.shape != (X.shape[1],)
            or std.shape != mean.shape or not np.all(np.isfinite(X))
            or not np.all(np.isfinite(mean)) or not np.all(std > 0)):
        raise ValueError("X与预处理参数不匹配")
    return (X - mean) / std


def sigmoid(logits: np.ndarray) -> np.ndarray:
    if not isinstance(logits, np.ndarray) or not np.all(np.isfinite(logits)):
        raise ValueError("logits必须有限")
    result = np.empty_like(logits, dtype=float); positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive]); result[~positive] = exp_value / (1.0 + exp_value)
    return result


def loss_and_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                      bias: float, *, l2: float) -> tuple[float, np.ndarray, float]:
    if (X.ndim != 2 or y.shape != (len(X),) or weights.shape != (X.shape[1],)
            or len(X) == 0 or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y))
            or not np.all(np.isin(y, [0.0, 1.0])) or not np.all(np.isfinite(weights))
            or not np.isfinite(bias) or not np.isfinite(l2) or l2 < 0):
        raise ValueError("训练输入或参数无效")
    logits = X @ weights + float(bias)
    data_loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
    probabilities = sigmoid(logits); error = probabilities - y
    gradient_weights = X.T @ error / len(X) + float(l2) * weights
    gradient_bias = float(np.mean(error))
    return float(data_loss + 0.5 * float(l2) * (weights @ weights)), gradient_weights, gradient_bias


def train(X: np.ndarray, y: np.ndarray, *, learning_rate: float = 0.2,
          steps: int = 1500, l2: float = 0.02) -> dict[str, np.ndarray | float]:
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
            or isinstance(learning_rate, bool) or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate和steps无效")
    weights = np.zeros(X.shape[1], dtype=float); bias = 0.0
    history = []
    for _ in range(steps):
        loss, grad_weights, grad_bias = loss_and_gradient(X, y, weights, bias, l2=l2)
        history.append(loss); weights = weights - float(learning_rate) * grad_weights; bias -= float(learning_rate) * grad_bias
    final_loss = loss_and_gradient(X, y, weights, bias, l2=l2)[0]
    return {"weights": weights, "bias": float(bias), "loss_history": np.asarray([*history, final_loss])}


def predict_probability(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    if X.ndim != 2 or weights.shape != (X.shape[1],) or not np.all(np.isfinite(X)):
        raise ValueError("预测形状错误")
    return sigmoid(X @ weights + float(bias))


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if y.shape != probabilities.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("指标输入形状错误")
    prediction = (probabilities >= 0.5).astype(int); truth = y.astype(int)
    tp = int(np.count_nonzero((prediction == 1) & (truth == 1)))
    fp = int(np.count_nonzero((prediction == 1) & (truth == 0)))
    fn = int(np.count_nonzero((prediction == 0) & (truth == 1)))
    accuracy = float(np.mean(prediction == truth)); denominator = 2 * tp + fp + fn
    return {"accuracy": accuracy, "f1": 0.0 if denominator == 0 else 2 * tp / denominator}


def save_model(path: Path, mean: np.ndarray, std: np.ndarray,
               weights: np.ndarray, bias: float) -> None:
    if mean.ndim != 1 or std.shape != mean.shape or weights.shape != mean.shape or not np.all(std > 0):
        raise ValueError("模型形状错误")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, mean=mean, std=std, weights=weights, bias=np.asarray(bias))


def load_model(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != MODEL_KEYS:
            raise ValueError(f"模型必须只包含{sorted(MODEL_KEYS)}")
        mean = archive["mean"].astype(float, copy=True); std = archive["std"].astype(float, copy=True)
        weights = archive["weights"].astype(float, copy=True); bias_array = archive["bias"].astype(float, copy=True)
    if (mean.ndim != 1 or std.shape != mean.shape or weights.shape != mean.shape
            or bias_array.shape != () or not np.all(np.isfinite(mean)) or not np.all(std > 0)
            or not np.all(np.isfinite(weights)) or not np.isfinite(bias_array)):
        raise ValueError("模型参数无效")
    return mean, std, weights, float(bias_array)


def save_predictions(path: Path, ids: list[str], probabilities: np.ndarray) -> None:
    if probabilities.shape != (len(ids),) or not np.all(np.isfinite(probabilities)):
        raise ValueError("预测形状或数值错误")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n"); writer.writerow(["sample_id", "probability", "label"])
        for sample_id, probability in zip(ids, probabilities, strict=True):
            writer.writerow([sample_id, f"{probability:.6f}", int(probability >= 0.5)])


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    if list(metrics) != ["accuracy", "f1"] or not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("指标格式错误")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={value:.6f}\n" for key, value in metrics.items()), encoding="utf-8", newline="\n")


def run(train_path: Path, validation_path: Path, test_path: Path,
        model_path: Path, predictions_path: Path, metrics_path: Path) -> None:
    _, X_train, y_train = load_csv(train_path, has_label=True)
    _, X_validation, y_validation = load_csv(validation_path, has_label=True)
    test_ids, X_test, _ = load_csv(test_path, has_label=False)
    assert y_train is not None and y_validation is not None
    mean, std = fit_preprocessor(X_train)
    train_scaled = transform(X_train, mean, std); validation_scaled = transform(X_validation, mean, std)
    model = train(train_scaled, y_train)
    metrics = classification_metrics(y_validation, predict_probability(validation_scaled, model["weights"], model["bias"]))
    save_model(model_path, mean, std, model["weights"], model["bias"])
    loaded_mean, loaded_std, loaded_weights, loaded_bias = load_model(model_path)
    probabilities = predict_probability(transform(X_test, loaded_mean, loaded_std), loaded_weights, loaded_bias)
    save_predictions(predictions_path, test_ids, probabilities); save_metrics(metrics_path, metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("train", "validation", "test", "model", "predictions", "metrics"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.train, args.validation, args.test, args.model, args.predictions, args.metrics)
    except (OSError, ValueError) as exc:
        print(f"第3题失败：{exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
