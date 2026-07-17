"""模拟卷2第3题参考实现：训练隔离、PCA、kNN、选参和最终重训。"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


FEATURES = ["feature_1", "feature_2", "feature_3"]
MODEL_KEYS = {"mean", "std", "components", "train_projection", "train_labels", "selected_k"}


def load_csv(path: Path, *, has_label: bool) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    header = ["sample_id", *FEATURES] + (["label"] if has_label else [])
    ids: list[str] = []
    rows: list[list[float]] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != header:
            raise ValueError(f"{path.name}表头错误")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[name] is None for name in header):
                raise ValueError(f"{path.name}第{line_number}行列数错误")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in ids:
                raise ValueError(f"{path.name}含空或重复sample_id")
            ids.append(sample_id)
            values = []
            for feature in FEATURES:
                token = row[feature].strip()
                if token == "":
                    values.append(np.nan)
                    continue
                try:
                    value = float(token)
                except ValueError as error:
                    raise ValueError(f"{path.name}第{line_number}行特征非法") from error
                if not np.isfinite(value):
                    raise ValueError("显式nan/inf非法")
                values.append(value)
            rows.append(values)
            if has_label:
                token = row["label"].strip()
                try:
                    value = float(token)
                except ValueError as error:
                    raise ValueError(f"{path.name}标签非法") from error
                if not np.isfinite(value) or not value.is_integer():
                    raise ValueError("标签必须是有限整数")
                labels.append(int(value))
    if not rows:
        raise ValueError(f"{path.name}不能只有表头")
    X = np.asarray(rows, dtype=float).reshape(len(rows), len(FEATURES))
    y = np.asarray(labels, dtype=int) if has_label else None
    return ids, X, y


def fit_preprocessor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(FEATURES):
        raise ValueError("X形状必须是(n,3)")
    if np.any(np.isinf(X)):
        raise ValueError("X不能含inf")
    if np.any(np.all(np.isnan(X), axis=0)):
        raise ValueError("拟合数据不能有整列缺失")
    mean = np.nanmean(X, axis=0)
    filled = np.where(np.isnan(X), mean[None, :], X)
    std = filled.std(axis=0)
    std[std == 0.0] = 1.0
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("预处理统计量必须有限")
    return mean, std


def transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError("X形状必须是(n,3)")
    if mean.shape != (len(FEATURES),) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("mean/std形状或数值错误")
    if np.any(np.isinf(X)):
        raise ValueError("X不能含inf")
    result = (np.where(np.isnan(X), mean[None, :], X) - mean) / std
    if not np.all(np.isfinite(result)):
        raise ValueError("变换结果必须有限")
    return result


def fit_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    if X_scaled.ndim != 2 or not np.all(np.isfinite(X_scaled)):
        raise ValueError("X_scaled必须是有限二维数组")
    if not isinstance(n_components, int) or not 1 <= n_components <= X_scaled.shape[1]:
        raise ValueError("n_components越界")
    covariance = X_scaled.T @ X_scaled / X_scaled.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    values = eigenvalues[order]
    components = eigenvectors[:, order].T
    for row in components:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0:
            row *= -1.0
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(components)):
        raise ValueError("PCA结果必须有限")
    return components, values


def project(X_scaled: np.ndarray, components: np.ndarray) -> np.ndarray:
    if X_scaled.ndim != 2 or components.ndim != 2 or X_scaled.shape[1] != components.shape[1]:
        raise ValueError("PCA投影形状不匹配")
    result = X_scaled @ components.T
    if not np.all(np.isfinite(result)):
        raise ValueError("投影必须有限")
    return result


def knn_predict(
    train_projection: np.ndarray,
    train_labels: np.ndarray,
    query_projection: np.ndarray,
    k: int,
) -> np.ndarray:
    if (
        train_projection.ndim != 2
        or query_projection.ndim != 2
        or train_projection.shape[1] != query_projection.shape[1]
        or train_labels.shape != (train_projection.shape[0],)
        or not np.all(np.isfinite(train_projection))
        or not np.all(np.isfinite(query_projection))
    ):
        raise ValueError("kNN输入形状或数值错误")
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or not 1 <= int(k) <= len(train_projection):
        raise ValueError("k必须是有效正整数")
    labels = np.unique(train_labels)
    predictions = np.empty(len(query_projection), dtype=train_labels.dtype)
    row_indices = np.arange(len(train_projection))
    for query_index, query in enumerate(query_projection):
        distances = np.sum((train_projection - query) ** 2, axis=1)
        neighbors = np.lexsort((row_indices, distances))[: int(k)]
        choices = []
        for label in labels:
            mask = train_labels[neighbors] == label
            choices.append((-int(mask.sum()), float(distances[neighbors][mask].sum()), label))
        predictions[query_index] = min(choices)[2]
    return predictions


def fit_pipeline(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y形状错误")
    mean, std = fit_preprocessor(X)
    scaled = transform(X, mean, std)
    components, eigenvalues = fit_pca(scaled, 2)
    return {
        "mean": mean,
        "std": std,
        "components": components,
        "train_projection": project(scaled, components),
        "train_labels": y.copy(),
        "eigenvalues": eigenvalues,
    }


def select_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    candidates: tuple[int, ...] = (1, 3, 5),
) -> tuple[int, float, list[tuple[int, float]]]:
    model = fit_pipeline(X_train, y_train)
    validation_projection = project(
        transform(X_validation, model["mean"], model["std"]), model["components"]
    )
    records = []
    for k in candidates:
        predictions = knn_predict(
            model["train_projection"], model["train_labels"], validation_projection, k
        )
        records.append((k, float(np.mean(predictions == y_validation))))
    selected_k, best_accuracy = min(records, key=lambda item: (-item[1], item[0]))
    return selected_k, best_accuracy, records


def save_model(path: Path, model: dict[str, np.ndarray], selected_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=model["mean"],
        std=model["std"],
        components=model["components"],
        train_projection=model["train_projection"],
        train_labels=model["train_labels"],
        selected_k=np.asarray(selected_k),
    )


def load_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != MODEL_KEYS:
            raise ValueError("模型键集合错误")
        model = {key: archive[key].copy() for key in archive.files}
    n = len(model["train_labels"])
    if (
        model["mean"].shape != (3,)
        or model["std"].shape != (3,)
        or model["components"].shape != (2, 3)
        or model["train_projection"].shape != (n, 2)
        or model["train_labels"].shape != (n,)
        or model["selected_k"].shape != ()
        or not np.all(model["std"] > 0)
        or not all(np.all(np.isfinite(value)) for value in model.values())
    ):
        raise ValueError("模型形状或数值错误")
    k = int(model["selected_k"])
    if not 1 <= k <= n:
        raise ValueError("模型selected_k错误")
    return model


def run(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    model_path: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> None:
    _, X_train, y_train = load_csv(train_path, has_label=True)
    _, X_validation, y_validation = load_csv(validation_path, has_label=True)
    test_ids, X_test, _ = load_csv(test_path, has_label=False)
    assert y_train is not None and y_validation is not None
    selected_k, validation_accuracy, _ = select_k(
        X_train, y_train, X_validation, y_validation
    )
    X_development = np.concatenate([X_train, X_validation], axis=0)
    y_development = np.concatenate([y_train, y_validation], axis=0)
    final_model = fit_pipeline(X_development, y_development)
    save_model(model_path, final_model, selected_k)
    restored = load_model(model_path)
    test_projection = project(
        transform(X_test, restored["mean"], restored["std"]), restored["components"]
    )
    predictions = knn_predict(
        restored["train_projection"],
        restored["train_labels"],
        test_projection,
        int(restored["selected_k"]),
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", "label"])
        writer.writerows(zip(test_ids, predictions.tolist(), strict=True))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        f"selected_k={selected_k}\nvalidation_accuracy={validation_accuracy:.6f}\n",
        encoding="utf-8",
        newline="",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.train, args.validation, args.test, args.model, args.predictions, args.metrics)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
