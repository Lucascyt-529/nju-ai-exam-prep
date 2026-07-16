"""参考实现：从CSV读取到参数恢复和结果保存的完整预处理程序。"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = ["feature_1", "feature_2"]
TRAIN_HEADER = ["sample_id", *FEATURE_COLUMNS, "label"]
TEST_HEADER = ["sample_id", *FEATURE_COLUMNS]
PARAMETER_KEYS = {"fill_values", "means", "scales"}


def _parse_optional_float(text: str, line_number: int, column: str) -> float:
    value = text.strip()
    if value == "":
        return np.nan
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"第 {line_number} 行 {column} 不是合法浮点数") from exc
    if not np.isfinite(number):
        raise ValueError(f"第 {line_number} 行 {column} 不能是 NaN 或无穷大")
    return number


def load_feature_csv(
    path: Path, *, has_label: bool
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """读取固定两列特征；缺失特征保留为 NaN。"""
    expected_header = TRAIN_HEADER if has_label else TEST_HEADER
    sample_ids: list[str] = []
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_header:
            raise ValueError(f"表头必须按以下顺序出现: {expected_header}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第 {line_number} 行字段数量与表头不一致")
            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"第 {line_number} 行 sample_id 为空")
            if sample_id in seen_ids:
                raise ValueError(f"第 {line_number} 行 sample_id 重复: {sample_id}")
            features = [
                _parse_optional_float(row[column], line_number, column)
                for column in FEATURE_COLUMNS
            ]
            if has_label:
                try:
                    label = int(row["label"].strip())
                except ValueError as exc:
                    raise ValueError(f"第 {line_number} 行 label 不是合法整数") from exc
                labels.append(label)
            sample_ids.append(sample_id)
            feature_rows.append(features)
            seen_ids.add(sample_id)

    if not feature_rows:
        raise ValueError("CSV 中没有数据行")
    X = np.asarray(feature_rows, dtype=float)
    y = np.asarray(labels, dtype=int) if has_label else None
    return sample_ids, X, y


def _validate_matrix(X: np.ndarray, *, allow_nan: bool) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维 NumPy 数组")
    if np.any(np.isinf(X)):
        raise ValueError("X 不能包含无穷大")
    if not allow_nan and np.any(np.isnan(X)):
        raise ValueError("X 不能包含 NaN")


def _validate_parameters(
    fill_values: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> None:
    shapes = {fill_values.shape, means.shape, scales.shape}
    if fill_values.ndim != 1 or len(shapes) != 1 or fill_values.size == 0:
        raise ValueError("三个参数必须是长度一致的非空一维数组")
    if not all(np.all(np.isfinite(values)) for values in (fill_values, means, scales)):
        raise ValueError("预处理参数必须只包含有限数值")
    if np.any(scales <= 0):
        raise ValueError("scales 必须全部大于0")


def fit_preprocessor(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回训练集拟合的 fill_values、means 和安全 scales。"""
    _validate_matrix(X_train, allow_nan=True)
    all_missing = np.all(np.isnan(X_train), axis=0)
    if np.any(all_missing):
        raise ValueError(f"训练集存在整列缺失: {np.flatnonzero(all_missing).tolist()}")
    fill_values = np.nanmean(X_train, axis=0)
    filled = np.where(np.isnan(X_train), fill_values, X_train)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales[scales == 0] = 1.0
    return fill_values, means, scales


def transform_features(
    X: np.ndarray,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """使用已有参数填补并标准化，不重新拟合。"""
    _validate_matrix(X, allow_nan=True)
    _validate_parameters(fill_values, means, scales)
    if X.shape[1] != fill_values.shape[0]:
        raise ValueError("X 的特征数与预处理参数不一致")
    filled = np.where(np.isnan(X), fill_values, X)
    transformed = (filled - means) / scales
    if not np.all(np.isfinite(transformed)):
        raise ValueError("变换结果包含非有限数值")
    return transformed


def save_parameters(
    path: Path,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> None:
    """把三个参数数组保存到一个 .npz 文件。"""
    _validate_parameters(fill_values, means, scales)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, fill_values=fill_values, means=means, scales=scales)


def load_parameters(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 .npz 恢复并验证三个参数数组。"""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PARAMETER_KEYS:
            raise ValueError(f"参数文件必须包含且只包含: {sorted(PARAMETER_KEYS)}")
        fill_values = archive["fill_values"].astype(float, copy=True)
        means = archive["means"].astype(float, copy=True)
        scales = archive["scales"].astype(float, copy=True)
    _validate_parameters(fill_values, means, scales)
    return fill_values, means, scales


def save_transformed_csv(
    path: Path, sample_ids: list[str], X_transformed: np.ndarray
) -> None:
    """按固定表头和6位小数保存测试特征。"""
    _validate_matrix(X_transformed, allow_nan=False)
    if X_transformed.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("本任务要求恰好两个输出特征")
    if len(sample_ids) != X_transformed.shape[0] or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids 必须唯一且与样本数一致")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", *FEATURE_COLUMNS])
        for sample_id, row in zip(sample_ids, X_transformed, strict=True):
            writer.writerow([sample_id, *(f"{value:.6f}" for value in row)])


def run_pipeline(
    train_path: Path, test_path: Path, output_path: Path, params_path: Path
) -> None:
    """组织训练拟合、参数恢复和测试变换。"""
    _, X_train, y_train = load_feature_csv(train_path, has_label=True)
    test_ids, X_test, _ = load_feature_csv(test_path, has_label=False)
    if y_train is None:
        raise RuntimeError("训练数据缺少标签")

    fill_values, means, scales = fit_preprocessor(X_train)
    save_parameters(params_path, fill_values, means, scales)
    loaded_fill, loaded_means, loaded_scales = load_parameters(params_path)
    X_test_transformed = transform_features(
        X_test, loaded_fill, loaded_means, loaded_scales
    )
    save_transformed_csv(output_path, test_ids, X_test_transformed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_pipeline(args.train, args.test, args.output, args.params)
    except (OSError, ValueError) as exc:
        print(f"数据准备失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
