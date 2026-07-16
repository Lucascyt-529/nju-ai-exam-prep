"""参考实现：完整的对数几率回归CSV分类程序。"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


FEATURES = ["feature_1", "feature_2"]
TRAIN_HEADER = ["sample_id", *FEATURES, "label"]
TEST_HEADER = ["sample_id", *FEATURES]
BUNDLE_KEYS = {"means", "scales", "weights", "bias", "threshold"}
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]


def load_classification_csv(path: Path, *, has_label: bool) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    expected = TRAIN_HEADER if has_label else TEST_HEADER
    ids: list[str] = []
    rows: list[list[float]] = []
    labels: list[int] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected:
            raise ValueError(f"表头必须按以下顺序出现: {expected}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第 {line_number} 行字段数量错误")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"第 {line_number} 行 sample_id 为空或重复")
            try:
                features = [float(row[name]) for name in FEATURES]
                label = int(row["label"]) if has_label else None
            except ValueError as exc:
                raise ValueError(f"第 {line_number} 行包含非法数值") from exc
            values = features + ([] if label is None else [label])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"第 {line_number} 行包含非有限数值")
            if label is not None and label not in (0, 1):
                raise ValueError(f"第 {line_number} 行 label 必须是0或1")
            ids.append(sample_id)
            rows.append(features)
            if label is not None:
                labels.append(label)
            seen.add(sample_id)
    if not rows:
        raise ValueError("CSV中没有数据行")
    return ids, np.asarray(rows, dtype=float), (np.asarray(labels, dtype=int) if has_label else None)


def _validate_X(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X必须只包含有限值")


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_X(X_train)
    means = X_train.mean(axis=0)
    scales = X_train.std(axis=0)
    scales[scales == 0] = 1.0
    return means, scales


def transform_standardizer(X: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    _validate_X(X)
    if means.ndim != 1 or scales.ndim != 1 or means.shape != scales.shape or means.shape[0] != X.shape[1]:
        raise ValueError("标准化参数形状与特征数不一致")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("标准化参数非法")
    return (X - means) / scales


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def fit_logistic_regression(
    X: np.ndarray, y: np.ndarray, *, learning_rate: float, n_steps: int, l2: float
) -> tuple[np.ndarray, float, np.ndarray]:
    _validate_X(X)
    if y.ndim != 1 or y.shape[0] != X.shape[0] or not np.all((y == 0) | (y == 1)):
        raise ValueError("y必须是一维0/1标签")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("学习率和迭代次数非法")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2必须是非负有限数值")
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    losses = np.empty(n_steps + 1)
    for step in range(n_steps + 1):
        logits = X @ weights + bias
        losses[step] = np.mean(np.logaddexp(0, logits) - y * logits) + 0.5 * l2 * (weights @ weights)
        if step == n_steps:
            break
        errors = _sigmoid(logits) - y
        weights -= learning_rate * (X.T @ errors / X.shape[0] + l2 * weights)
        bias -= learning_rate * float(errors.mean())
    if not np.all(np.isfinite(losses)):
        raise FloatingPointError("训练发散")
    return weights, bias, losses


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    _validate_X(X)
    if weights.ndim != 1 or weights.shape[0] != X.shape[1] or not np.all(np.isfinite(weights)) or not np.isfinite(bias):
        raise ValueError("模型参数非法")
    return _sigmoid(X @ weights + bias)


def _auc(y: np.ndarray, probabilities: np.ndarray) -> float:
    positives = np.count_nonzero(y == 1)
    negatives = np.count_nonzero(y == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC要求验证集同时含正负类")
    order = np.argsort(probabilities, kind="stable")
    sorted_scores = probabilities[order]
    ranks = np.empty(y.size, dtype=float)
    position = 0
    while position < y.size:
        end = position + 1
        while end < y.size and sorted_scores[end] == sorted_scores[position]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2
        position = end
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def validation_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    if y.ndim != 1 or probabilities.ndim != 1 or y.shape != probabilities.shape or y.size == 0:
        raise ValueError("标签与概率必须是形状一致的非空一维数组")
    if not np.all((y == 0) | (y == 1)) or not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("标签或概率非法")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold必须位于(0,1)")
    labels = (probabilities >= threshold).astype(int)
    tp = np.count_nonzero((y == 1) & (labels == 1))
    fp = np.count_nonzero((y == 0) & (labels == 1))
    fn = np.count_nonzero((y == 1) & (labels == 0))
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {
        "accuracy": float(np.mean(labels == y)),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "auc": _auc(y, probabilities),
    }


def save_bundle(
    path: Path, means: np.ndarray, scales: np.ndarray,
    weights: np.ndarray, bias: float, threshold: float
) -> None:
    if means.ndim != 1 or scales.shape != means.shape or weights.shape != means.shape:
        raise ValueError("模型包参数形状不一致")
    if not all(np.all(np.isfinite(value)) for value in (means, scales, weights)) or np.any(scales <= 0):
        raise ValueError("模型包包含非法数组")
    if not np.isfinite(bias) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("截距或阈值非法")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, means=means, scales=scales, weights=weights,
                 bias=np.asarray(bias), threshold=np.asarray(threshold))


def load_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != BUNDLE_KEYS:
            raise ValueError(f"模型包必须包含且只包含: {sorted(BUNDLE_KEYS)}")
        means = archive["means"].astype(float, copy=True)
        scales = archive["scales"].astype(float, copy=True)
        weights = archive["weights"].astype(float, copy=True)
        bias_array = archive["bias"].astype(float, copy=True)
        threshold_array = archive["threshold"].astype(float, copy=True)
    if means.ndim != 1 or scales.shape != means.shape or weights.shape != means.shape or bias_array.shape != () or threshold_array.shape != ():
        raise ValueError("模型包参数形状错误")
    bias, threshold = float(bias_array), float(threshold_array)
    if not all(np.all(np.isfinite(value)) for value in (means, scales, weights)) or np.any(scales <= 0) or not np.isfinite(bias) or not 0 < threshold < 1:
        raise ValueError("模型包参数非法")
    return means, scales, weights, bias, threshold


def save_predictions(path: Path, sample_ids: list[str], probabilities: np.ndarray, labels: np.ndarray) -> None:
    if probabilities.ndim != 1 or labels.ndim != 1 or probabilities.shape != labels.shape or len(sample_ids) != probabilities.size:
        raise ValueError("预测字段长度或形状不一致")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", "probability", "label"])
        for sample_id, probability, label in zip(sample_ids, probabilities, labels, strict=True):
            writer.writerow([sample_id, f"{probability:.6f}", str(int(label))])


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    if list(metrics) != METRIC_KEYS or not all(np.isfinite(metrics[key]) for key in METRIC_KEYS):
        raise ValueError("指标键顺序或数值非法")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={metrics[key]:.6f}\n" for key in METRIC_KEYS), encoding="utf-8", newline="\n")


def run_pipeline(
    train_path: Path, validation_path: Path, test_path: Path,
    predictions_path: Path, metrics_path: Path, bundle_path: Path,
    *, learning_rate: float, n_steps: int, l2: float, threshold: float,
) -> None:
    _, X_train, y_train = load_classification_csv(train_path, has_label=True)
    _, X_validation, y_validation = load_classification_csv(validation_path, has_label=True)
    test_ids, X_test, _ = load_classification_csv(test_path, has_label=False)
    assert y_train is not None and y_validation is not None
    means, scales = fit_standardizer(X_train)
    train_scaled = transform_standardizer(X_train, means, scales)
    validation_scaled = transform_standardizer(X_validation, means, scales)
    weights, bias, _ = fit_logistic_regression(
        train_scaled, y_train, learning_rate=learning_rate, n_steps=n_steps, l2=l2
    )
    validation_probabilities = predict_proba(validation_scaled, weights, bias)
    metrics = validation_metrics(y_validation, validation_probabilities, threshold)
    save_bundle(bundle_path, means, scales, weights, bias, threshold)
    loaded_means, loaded_scales, loaded_weights, loaded_bias, loaded_threshold = load_bundle(bundle_path)
    test_scaled = transform_standardizer(X_test, loaded_means, loaded_scales)
    probabilities = predict_proba(test_scaled, loaded_weights, loaded_bias)
    labels = (probabilities >= loaded_threshold).astype(int)
    save_predictions(predictions_path, test_ids, probabilities, labels)
    save_metrics(metrics_path, metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_pipeline(
            args.train, args.validation, args.test, args.predictions, args.metrics,
            args.bundle, learning_rate=args.learning_rate, n_steps=args.steps,
            l2=args.l2, threshold=args.threshold,
        )
    except (OSError, ValueError) as exc:
        print(f"分类任务失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
