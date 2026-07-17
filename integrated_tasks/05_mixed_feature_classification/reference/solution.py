"""参考实现：混合数值/类别特征的训练与独立预测程序。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


NUMERIC_FEATURES = ("age", "monthly_spend")
CATEGORICAL_FEATURES = ("city", "device")
TRAIN_HEADER = ("sample_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "label")
TEST_HEADER = ("sample_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
ENCODER_KEYS = {
    "format_version",
    "numeric_features",
    "categorical_features",
    "vocabularies",
    "include_unknown",
}
MODEL_KEYS = {"fill_values", "means", "scales", "weights", "bias", "threshold"}
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "auc")


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_finite_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def load_mixed_classification_csv(
    path: Path, *, has_label: bool
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
    """严格读取混合CSV；数值空白保留为NaN，类别空白不允许。"""
    expected_header = list(TRAIN_HEADER if has_label else TEST_HEADER)
    sample_ids: list[str] = []
    numeric_rows: list[list[float]] = []
    categorical_rows: list[list[str]] = []
    labels: list[int] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_header:
            raise ValueError(f"表头必须按以下顺序出现: {expected_header}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第{line_number}行字段数量错误")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"第{line_number}行sample_id为空或重复")

            numeric_values: list[float] = []
            for feature in NUMERIC_FEATURES:
                text = row[feature].strip()
                if text == "":
                    numeric_values.append(float("nan"))
                    continue
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(
                        f"第{line_number}行{feature}不是数值或空白"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"第{line_number}行{feature}不能是NaN或无穷大")
                numeric_values.append(value)

            categorical_values = [row[name].strip() for name in CATEGORICAL_FEATURES]
            if any(value == "" for value in categorical_values):
                raise ValueError(f"第{line_number}行类别特征不能为空")

            if has_label:
                label_text = row["label"].strip()
                if label_text not in {"0", "1"}:
                    raise ValueError(f"第{line_number}行label必须是0或1")
                labels.append(int(label_text))

            sample_ids.append(sample_id)
            numeric_rows.append(numeric_values)
            categorical_rows.append(categorical_values)
            seen_ids.add(sample_id)

    if not sample_ids:
        raise ValueError("CSV中没有数据行")
    y = np.asarray(labels, dtype=int) if has_label else None
    return (
        sample_ids,
        np.asarray(numeric_rows, dtype=float),
        np.asarray(categorical_rows, dtype=str),
        y,
    )


def _validate_numeric_matrix(X_numeric: np.ndarray, *, allow_nan: bool) -> None:
    if (
        not isinstance(X_numeric, np.ndarray)
        or not np.issubdtype(X_numeric.dtype, np.number)
        or np.issubdtype(X_numeric.dtype, np.bool_)
        or X_numeric.ndim != 2
        or X_numeric.shape[0] == 0
        or X_numeric.shape[1] != len(NUMERIC_FEATURES)
        or np.any(np.isinf(X_numeric))
        or (not allow_nan and np.any(np.isnan(X_numeric)))
    ):
        suffix = "，允许NaN" if allow_nan else "，不允许NaN"
        raise ValueError(f"数值特征必须是形状(n,{len(NUMERIC_FEATURES)})的有限矩阵{suffix}")


def _validate_categorical_matrix(X_categorical: np.ndarray) -> None:
    if (
        not isinstance(X_categorical, np.ndarray)
        or X_categorical.ndim != 2
        or X_categorical.shape[0] == 0
        or X_categorical.shape[1] != len(CATEGORICAL_FEATURES)
    ):
        raise ValueError(
            f"类别特征必须是形状(n,{len(CATEGORICAL_FEATURES)})的非空二维数组"
        )
    if any(not isinstance(value, str) or not value.strip() for value in X_categorical.ravel()):
        raise ValueError("类别特征必须是非空字符串")


def fit_numeric_preprocessor(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只用训练集拟合均值填补、填补后的均值和安全标准差。"""
    _validate_numeric_matrix(X_train, allow_nan=True)
    all_missing = np.all(np.isnan(X_train), axis=0)
    if np.any(all_missing):
        raise ValueError(f"数值特征整列缺失: {np.flatnonzero(all_missing).tolist()}")
    fill_values = np.nanmean(X_train, axis=0)
    filled = np.where(np.isnan(X_train), fill_values, X_train)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales[scales == 0.0] = 1.0
    return fill_values, means, scales


def _validate_numeric_state(
    fill_values: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> None:
    expected_shape = (len(NUMERIC_FEATURES),)
    if any(
        not isinstance(value, np.ndarray)
        or value.shape != expected_shape
        or not np.issubdtype(value.dtype, np.number)
        or not np.all(np.isfinite(value))
        for value in (fill_values, means, scales)
    ):
        raise ValueError("数值预处理参数必须是与数值特征同长度的有限向量")
    if np.any(scales <= 0):
        raise ValueError("scales必须全部大于0")


def transform_numeric(
    X_numeric: np.ndarray,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    _validate_numeric_matrix(X_numeric, allow_nan=True)
    _validate_numeric_state(fill_values, means, scales)
    filled = np.where(np.isnan(X_numeric), fill_values, X_numeric)
    return (filled - means) / scales


def fit_vocabularies(X_train: np.ndarray) -> tuple[tuple[str, ...], ...]:
    _validate_categorical_matrix(X_train)
    return tuple(
        tuple(sorted(set(X_train[:, column].tolist())))
        for column in range(X_train.shape[1])
    )


def _validate_vocabularies(vocabularies: tuple[tuple[str, ...], ...]) -> None:
    if not isinstance(vocabularies, tuple) or len(vocabularies) != len(
        CATEGORICAL_FEATURES
    ):
        raise ValueError("类别词表数量与类别特征数不一致")
    for vocabulary in vocabularies:
        if (
            not isinstance(vocabulary, tuple)
            or not vocabulary
            or any(not isinstance(value, str) or not value for value in vocabulary)
            or tuple(sorted(set(vocabulary))) != vocabulary
        ):
            raise ValueError("每个类别词表必须是非空、唯一、升序字符串元组")


def transform_one_hot(
    X_categorical: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    """每个类别特征最后一列固定为未知类别桶。"""
    _validate_categorical_matrix(X_categorical)
    _validate_vocabularies(vocabularies)
    widths = [len(vocabulary) + 1 for vocabulary in vocabularies]
    result = np.zeros((X_categorical.shape[0], sum(widths)), dtype=float)
    offset = 0
    for column, (vocabulary, width) in enumerate(zip(vocabularies, widths, strict=True)):
        mapping = {value: index for index, value in enumerate(vocabulary)}
        unknown_index = len(vocabulary)
        for row, value in enumerate(X_categorical[:, column]):
            result[row, offset + mapping.get(value, unknown_index)] = 1.0
        offset += width
    return result


def build_design_matrix(
    X_numeric: np.ndarray,
    X_categorical: np.ndarray,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    vocabularies: tuple[tuple[str, ...], ...],
) -> np.ndarray:
    if X_numeric.shape[0] != X_categorical.shape[0]:
        raise ValueError("数值特征和类别特征的样本数不一致")
    numeric_scaled = transform_numeric(X_numeric, fill_values, means, scales)
    categorical_one_hot = transform_one_hot(X_categorical, vocabularies)
    return np.concatenate((numeric_scaled, categorical_one_hot), axis=1)


def save_encoder_metadata(
    path: Path, vocabularies: tuple[tuple[str, ...], ...]
) -> None:
    _validate_vocabularies(vocabularies)
    payload = {
        "format_version": 1,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "vocabularies": [list(vocabulary) for vocabulary in vocabularies],
        "include_unknown": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_encoder_metadata(path: Path) -> tuple[tuple[str, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != ENCODER_KEYS:
        raise ValueError("编码器JSON键集合错误")
    if not isinstance(payload["format_version"], int) or isinstance(
        payload["format_version"], bool
    ) or payload["format_version"] != 1:
        raise ValueError("不支持的编码器格式版本")
    if payload["numeric_features"] != list(NUMERIC_FEATURES):
        raise ValueError("数值特征名称或顺序错误")
    if payload["categorical_features"] != list(CATEGORICAL_FEATURES):
        raise ValueError("类别特征名称或顺序错误")
    if payload["include_unknown"] is not True:
        raise ValueError("编码器必须启用未知类别桶")
    raw_vocabularies = payload["vocabularies"]
    if not isinstance(raw_vocabularies, list) or any(
        not isinstance(vocabulary, list) for vocabulary in raw_vocabularies
    ):
        raise ValueError("vocabularies必须是二维列表")
    try:
        vocabularies = tuple(tuple(vocabulary) for vocabulary in raw_vocabularies)
    except TypeError as exc:
        raise ValueError("vocabularies结构错误") from exc
    _validate_vocabularies(vocabularies)
    return vocabularies


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or 0 in X.shape
        or not np.all(np.isfinite(X))
    ):
        raise ValueError("X必须是非空有限二维数组")
    if (
        not isinstance(y, np.ndarray)
        or y.shape != (X.shape[0],)
        or not np.all(np.isin(y, [0, 1]))
    ):
        raise ValueError("y必须是与X样本数一致的一维0/1标签")
    if not _is_finite_scalar(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate必须是正有限数值")
    if not _is_integer(n_steps) or n_steps < 1:
        raise ValueError("n_steps必须是正整数")
    if not _is_finite_scalar(l2) or l2 < 0:
        raise ValueError("l2必须是非负有限数值")

    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0
    losses = np.empty(int(n_steps) + 1)
    for step in range(int(n_steps) + 1):
        logits = X @ weights + bias
        losses[step] = np.mean(np.logaddexp(0.0, logits) - y * logits)
        losses[step] += 0.5 * float(l2) * float(weights @ weights)
        if step == int(n_steps):
            break
        errors = _sigmoid(logits) - y
        weights -= float(learning_rate) * (
            X.T @ errors / X.shape[0] + float(l2) * weights
        )
        bias -= float(learning_rate) * float(np.mean(errors))
    if not np.all(np.isfinite(losses)):
        raise FloatingPointError("训练发散")
    return weights, bias, losses


def predict_probabilities(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or not np.all(np.isfinite(X))
        or not isinstance(weights, np.ndarray)
        or weights.shape != (X.shape[1],)
        or not np.all(np.isfinite(weights))
        or not _is_finite_scalar(bias)
    ):
        raise ValueError("设计矩阵或模型参数非法")
    return _sigmoid(X @ weights + float(bias))


def _auc(y: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("AUC要求验证集同时含正负类")
    order = np.argsort(probabilities, kind="stable")
    sorted_scores = probabilities[order]
    ranks = np.empty(y.size, dtype=float)
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def validation_metrics(
    y: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(probabilities, np.ndarray)
        or y.ndim != 1
        or probabilities.shape != y.shape
        or y.size == 0
        or not np.all(np.isin(y, [0, 1]))
        or not np.all(np.isfinite(probabilities))
        or not np.all((probabilities >= 0) & (probabilities <= 1))
    ):
        raise ValueError("验证标签与概率必须是同形状的合法一维数组")
    if not _is_finite_scalar(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold必须位于(0,1)")
    labels = (probabilities >= float(threshold)).astype(int)
    tp = int(np.sum((y == 1) & (labels == 1)))
    fp = int(np.sum((y == 0) & (labels == 1)))
    fn = int(np.sum((y == 1) & (labels == 0)))
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {
        "accuracy": float(np.mean(labels == y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(_auc(y, probabilities)),
    }


def save_model_bundle(
    path: Path,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold: float,
) -> None:
    _validate_numeric_state(fill_values, means, scales)
    if (
        not isinstance(weights, np.ndarray)
        or weights.ndim != 1
        or weights.size == 0
        or not np.all(np.isfinite(weights))
        or not _is_finite_scalar(bias)
        or not _is_finite_scalar(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("模型权重、截距或阈值非法")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(
            file,
            fill_values=fill_values,
            means=means,
            scales=scales,
            weights=weights,
            bias=np.asarray(float(bias)),
            threshold=np.asarray(float(threshold)),
        )


def load_model_bundle(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != MODEL_KEYS:
            raise ValueError(f"模型包必须包含且只包含: {sorted(MODEL_KEYS)}")
        fill_values = archive["fill_values"].astype(float, copy=True)
        means = archive["means"].astype(float, copy=True)
        scales = archive["scales"].astype(float, copy=True)
        weights = archive["weights"].astype(float, copy=True)
        bias_array = archive["bias"].astype(float, copy=True)
        threshold_array = archive["threshold"].astype(float, copy=True)
    _validate_numeric_state(fill_values, means, scales)
    if (
        weights.ndim != 1
        or weights.size == 0
        or not np.all(np.isfinite(weights))
        or bias_array.shape != ()
        or threshold_array.shape != ()
    ):
        raise ValueError("模型包数组形状或数值非法")
    bias = float(bias_array)
    threshold = float(threshold_array)
    if not np.isfinite(bias) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("模型包截距或阈值非法")
    return fill_values, means, scales, weights, bias, threshold


def _expected_design_width(vocabularies: tuple[tuple[str, ...], ...]) -> int:
    return len(NUMERIC_FEATURES) + sum(
        len(vocabulary) + 1 for vocabulary in vocabularies
    )


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    if list(metrics) != list(METRIC_KEYS) or not all(
        np.isfinite(metrics[key]) for key in METRIC_KEYS
    ):
        raise ValueError("指标键顺序或数值非法")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={metrics[key]:.6f}\n" for key in METRIC_KEYS),
        encoding="utf-8",
        newline="\n",
    )


def save_predictions(
    path: Path,
    sample_ids: list[str],
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> None:
    if (
        probabilities.ndim != 1
        or labels.shape != probabilities.shape
        or len(sample_ids) != probabilities.size
        or len(set(sample_ids)) != len(sample_ids)
        or not np.all(np.isfinite(probabilities))
        or not np.all((probabilities >= 0) & (probabilities <= 1))
        or not np.all(np.isin(labels, [0, 1]))
    ):
        raise ValueError("预测字段形状或数值非法")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", "probability", "label"])
        for sample_id, probability, label in zip(
            sample_ids, probabilities, labels, strict=True
        ):
            writer.writerow([sample_id, f"{probability:.6f}", str(int(label))])


def run_training(
    train_path: Path,
    validation_path: Path,
    encoder_path: Path,
    model_path: Path,
    metrics_path: Path,
    *,
    learning_rate: float,
    n_steps: int,
    l2: float,
    threshold: float,
) -> None:
    _, train_numeric, train_categorical, y_train = load_mixed_classification_csv(
        train_path, has_label=True
    )
    _, validation_numeric, validation_categorical, y_validation = (
        load_mixed_classification_csv(validation_path, has_label=True)
    )
    assert y_train is not None and y_validation is not None

    fill_values, means, scales = fit_numeric_preprocessor(train_numeric)
    vocabularies = fit_vocabularies(train_categorical)
    X_train = build_design_matrix(
        train_numeric,
        train_categorical,
        fill_values,
        means,
        scales,
        vocabularies,
    )
    weights, bias, _ = fit_logistic_regression(
        X_train,
        y_train,
        learning_rate=learning_rate,
        n_steps=n_steps,
        l2=l2,
    )
    X_validation = build_design_matrix(
        validation_numeric,
        validation_categorical,
        fill_values,
        means,
        scales,
        vocabularies,
    )
    validation_probabilities = predict_probabilities(X_validation, weights, bias)
    metrics = validation_metrics(y_validation, validation_probabilities, threshold)

    save_encoder_metadata(encoder_path, vocabularies)
    save_model_bundle(
        model_path, fill_values, means, scales, weights, bias, threshold
    )
    save_metrics(metrics_path, metrics)


def run_prediction(
    test_path: Path,
    encoder_path: Path,
    model_path: Path,
    predictions_path: Path,
) -> None:
    vocabularies = load_encoder_metadata(encoder_path)
    fill_values, means, scales, weights, bias, threshold = load_model_bundle(
        model_path
    )
    expected_width = _expected_design_width(vocabularies)
    if weights.shape != (expected_width,):
        raise ValueError(
            f"模型权重长度{weights.size}与编码后特征数{expected_width}不一致"
        )
    sample_ids, test_numeric, test_categorical, _ = load_mixed_classification_csv(
        test_path, has_label=False
    )
    X_test = build_design_matrix(
        test_numeric,
        test_categorical,
        fill_values,
        means,
        scales,
        vocabularies,
    )
    probabilities = predict_probabilities(X_test, weights, bias)
    labels = (probabilities >= threshold).astype(int)
    save_predictions(predictions_path, sample_ids, probabilities, labels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并保存预处理与模型状态")
    train_parser.add_argument("--train", type=Path, required=True)
    train_parser.add_argument("--validation", type=Path, required=True)
    train_parser.add_argument("--encoder", type=Path, required=True)
    train_parser.add_argument("--model", type=Path, required=True)
    train_parser.add_argument("--metrics", type=Path, required=True)
    train_parser.add_argument("--learning-rate", type=float, default=0.2)
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--l2", type=float, default=0.05)
    train_parser.add_argument("--threshold", type=float, default=0.5)

    predict_parser = subparsers.add_parser("predict", help="只从状态文件恢复并预测")
    predict_parser.add_argument("--test", type=Path, required=True)
    predict_parser.add_argument("--encoder", type=Path, required=True)
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--predictions", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            run_training(
                args.train,
                args.validation,
                args.encoder,
                args.model,
                args.metrics,
                learning_rate=args.learning_rate,
                n_steps=args.steps,
                l2=args.l2,
                threshold=args.threshold,
            )
        else:
            run_prediction(args.test, args.encoder, args.model, args.predictions)
    except (OSError, ValueError, FloatingPointError, json.JSONDecodeError) as exc:
        print(f"混合特征任务失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
