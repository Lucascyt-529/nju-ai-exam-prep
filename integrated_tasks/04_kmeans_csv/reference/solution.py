"""参考实现：无OJ环境下的CSV预处理、K-means选模、保存和预测。"""

import argparse
import csv
from numbers import Real
from pathlib import Path
import sys

import numpy as np


FEATURE_COLUMNS = ["feature_1", "feature_2"]
HEADER = ["sample_id", *FEATURE_COLUMNS]
MODEL_KEYS = {"mean", "std", "centers"}


def load_feature_csv(path: Path) -> tuple[list[str], np.ndarray]:
    sample_ids: list[str] = []
    rows: list[list[float]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != HEADER:
            raise ValueError(f"表头必须按以下顺序出现: {HEADER}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第 {line_number} 行字段数量错误")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"第 {line_number} 行sample_id为空或重复")
            values: list[float] = []
            for column in FEATURE_COLUMNS:
                text = row[column].strip()
                if not text:
                    values.append(float("nan"))
                    continue
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(f"第 {line_number} 行{column}不是合法数值") from exc
                if not np.isfinite(value):
                    raise ValueError(f"第 {line_number} 行{column}必须有限或留空")
                values.append(value)
            if all(np.isnan(values)):
                raise ValueError(f"第 {line_number} 行不能所有特征都缺失")
            sample_ids.append(sample_id); rows.append(values); seen.add(sample_id)
    if not rows:
        raise ValueError("CSV中没有数据行")
    return sample_ids, np.asarray(rows, dtype=float)


def _matrix_with_missing(X: np.ndarray, name: str) -> None:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not np.issubdtype(X.dtype, np.number) or np.any(np.isinf(X))):
        raise ValueError(f"{name}必须是非空数值二维数组，缺失值用NaN表示")


def fit_preprocessor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _matrix_with_missing(X, "X")
    if np.any(np.all(np.isnan(X), axis=0)):
        raise ValueError("训练集不能包含整列缺失")
    mean = np.nanmean(X.astype(float), axis=0)
    filled = np.where(np.isnan(X), mean[None, :], X)
    std = np.std(filled, axis=0)
    safe_std = np.where(std == 0.0, 1.0, std)
    return mean, safe_std


def transform_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    _matrix_with_missing(X, "X")
    if (not isinstance(mean, np.ndarray) or not isinstance(std, np.ndarray)
            or mean.shape != (X.shape[1],) or std.shape != mean.shape
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std))
            or np.any(std <= 0)):
        raise ValueError("mean/std必须是与特征数一致的有限一维数组，且std为正")
    filled = np.where(np.isnan(X), mean[None, :], X.astype(float))
    return (filled - mean) / std


def squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if (not isinstance(X, np.ndarray) or not isinstance(centers, np.ndarray)
            or X.ndim != 2 or centers.ndim != 2 or 0 in X.shape or 0 in centers.shape
            or X.shape[1] != centers.shape[1] or not np.all(np.isfinite(X))
            or not np.all(np.isfinite(centers))):
        raise ValueError("X和centers必须是特征数相同的非空有限二维数组")
    difference = X[:, None, :] - centers[None, :, :]
    return np.sum(difference * difference, axis=2)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return int(value)


def _seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("random_state必须是整数")
    return int(value)


def initialize_kmeans_plus_plus(X: np.ndarray, n_clusters: int, *,
                                random_state: int = 0) -> np.ndarray:
    k = _positive_int(n_clusters, "n_clusters")
    if not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape or not np.all(np.isfinite(X)):
        raise ValueError("X必须是非空有限二维数组")
    if k > np.unique(X, axis=0).shape[0]:
        raise ValueError("n_clusters不能超过不同样本点数量")
    rng = np.random.default_rng(_seed(random_state))
    centers = [X[int(rng.integers(len(X)))].astype(float).copy()]
    while len(centers) < k:
        closest = np.min(squared_distances(X, np.vstack(centers)), axis=1)
        total = float(np.sum(closest))
        if total <= 0:
            raise RuntimeError("剩余K-means++采样权重为0")
        centers.append(X[int(rng.choice(len(X), p=closest / total))].astype(float).copy())
    return np.vstack(centers)


def _canonicalize(centers: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = tuple(centers[:, column] for column in range(centers.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
    return centers[order], inverse[labels]


def fit_kmeans_once(X: np.ndarray, n_clusters: int, *, random_state: int = 0,
                    max_iterations: int = 100, tolerance: float = 1e-6) -> dict[str, object]:
    k = _positive_int(n_clusters, "n_clusters")
    limit = _positive_int(max_iterations, "max_iterations")
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance必须是非负有限实数")
    centers = initialize_kmeans_plus_plus(X, k, random_state=random_state)
    labels = np.argmin(squared_distances(X, centers), axis=1)
    for iteration in range(1, limit + 1):
        new_centers = np.empty_like(centers)
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                new_centers[cluster] = np.mean(members, axis=0)
            else:
                assigned = squared_distances(X, centers)[np.arange(len(X)), labels]
                new_centers[cluster] = X[int(np.argmax(assigned))]
        new_labels = np.argmin(squared_distances(X, new_centers), axis=1)
        movement = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        unchanged = np.array_equal(new_labels, labels)
        centers, labels = new_centers, new_labels
        if unchanged or movement <= float(tolerance):
            break
    centers, labels = _canonicalize(centers, labels)
    distances = squared_distances(X, centers)
    inertia = float(np.sum(distances[np.arange(len(X)), labels]))
    return {"centers": centers, "labels": labels, "inertia": inertia, "iterations": iteration}


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or not np.all(np.isfinite(X))
            or not isinstance(labels, np.ndarray) or labels.shape != (len(X),)):
        raise ValueError("X和labels形状不匹配")
    clusters = np.unique(labels)
    if not 2 <= len(clusters) < len(X):
        raise ValueError("轮廓系数要求2到n-1个非空簇")
    distances = np.sqrt(squared_distances(X, X))
    values = np.zeros(len(X), dtype=float)
    for index in range(len(X)):
        own = labels == labels[index]
        own_count = int(np.count_nonzero(own))
        if own_count == 1:
            values[index] = 0.0
            continue
        a = float(np.sum(distances[index, own]) / (own_count - 1))
        b = min(float(np.mean(distances[index, labels == cluster]))
                for cluster in clusters if cluster != labels[index])
        values[index] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(np.mean(values))


def select_kmeans_model(X: np.ndarray, candidate_ks: tuple[int, ...] | list[int], *,
                        n_init: int = 5, random_state: int = 0) -> dict[str, object]:
    attempts = _positive_int(n_init, "n_init")
    try:
        candidates = tuple(_positive_int(k, "candidate_k") for k in candidate_ks)
    except TypeError as exc:
        raise ValueError("candidate_ks必须是整数序列") from exc
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_ks必须非空且不能重复")
    if any(k < 2 or k >= len(X) for k in candidates):
        raise ValueError("每个candidate_k必须满足2 <= k < n_samples")
    root = np.random.default_rng(_seed(random_state))
    best_overall: dict[str, object] | None = None
    summaries: list[tuple[int, float, float]] = []
    for k in sorted(candidates):
        best_for_k: dict[str, object] | None = None
        for _ in range(attempts):
            seed = int(root.integers(0, np.iinfo(np.uint32).max))
            current = fit_kmeans_once(X, k, random_state=seed)
            if best_for_k is None or current["inertia"] < best_for_k["inertia"]:
                best_for_k = current
        assert best_for_k is not None
        score = silhouette_score(X, best_for_k["labels"])
        summaries.append((k, float(best_for_k["inertia"]), score))
        candidate = {**best_for_k, "n_clusters": k, "silhouette": score}
        if (best_overall is None or score > best_overall["silhouette"] + 1e-12
                or (np.isclose(score, best_overall["silhouette"]) and k < best_overall["n_clusters"])):
            best_overall = candidate
    assert best_overall is not None
    best_overall["candidate_summary"] = tuple(summaries)
    return best_overall


def save_model(path: Path, mean: np.ndarray, std: np.ndarray, centers: np.ndarray) -> None:
    if (mean.ndim != 1 or std.shape != mean.shape or centers.ndim != 2
            or centers.shape[1] != len(mean) or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std)) or not np.all(std > 0)
            or not np.all(np.isfinite(centers))):
        raise ValueError("模型参数形状或数值无效")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, mean=mean, std=std, centers=centers)


def load_model(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != MODEL_KEYS:
            raise ValueError(f"模型必须只包含: {sorted(MODEL_KEYS)}")
        mean = archive["mean"].astype(float, copy=True)
        std = archive["std"].astype(float, copy=True)
        centers = archive["centers"].astype(float, copy=True)
    if (mean.ndim != 1 or std.shape != mean.shape or centers.ndim != 2
            or centers.shape[1] != len(mean) or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std)) or not np.all(std > 0)
            or not np.all(np.isfinite(centers))):
        raise ValueError("模型参数形状或数值无效")
    return mean, std, centers


def save_assignments(path: Path, sample_ids: list[str], labels: np.ndarray) -> None:
    if labels.shape != (len(sample_ids),) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids与labels不匹配")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", "cluster"])
        writer.writerows(zip(sample_ids, labels.tolist(), strict=True))


def save_diagnostics(path: Path, model: dict[str, object]) -> None:
    content = (
        f"selected_k={model['n_clusters']}\n"
        f"silhouette={model['silhouette']:.6f}\n"
        f"inertia={model['inertia']:.6f}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def run_pipeline(train_path: Path, test_path: Path, model_path: Path,
                 assignments_path: Path, diagnostics_path: Path, *,
                 candidate_ks: tuple[int, ...] = (2, 3, 4),
                 n_init: int = 10, random_state: int = 0) -> None:
    _, X_train = load_feature_csv(train_path)
    test_ids, X_test = load_feature_csv(test_path)
    mean, std = fit_preprocessor(X_train)
    train_scaled = transform_features(X_train, mean, std)
    fitted = select_kmeans_model(
        train_scaled, candidate_ks, n_init=n_init, random_state=random_state
    )
    save_model(model_path, mean, std, fitted["centers"])
    loaded_mean, loaded_std, loaded_centers = load_model(model_path)
    test_scaled = transform_features(X_test, loaded_mean, loaded_std)
    labels = np.argmin(squared_distances(test_scaled, loaded_centers), axis=1)
    save_assignments(assignments_path, test_ids, labels)
    save_diagnostics(diagnostics_path, fitted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_pipeline(
            args.train, args.test, args.model, args.assignments, args.diagnostics,
            candidate_ks=tuple(args.candidate_k), n_init=args.n_init,
            random_state=args.random_state,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"K-means综合任务失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
