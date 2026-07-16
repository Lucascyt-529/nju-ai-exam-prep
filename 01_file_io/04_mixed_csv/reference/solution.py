"""参考实现：标准库 csv 读取混合类型数据并输出摘要。"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


EXPECTED_HEADER = ["sample_id", "name", "feature_1", "feature_2", "label"]


def load_mixed_dataset(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """返回 sample_ids、names、二维特征 X 和一维整数标签 y。"""
    sample_ids: list[str] = []
    names: list[str] = []
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"表头必须按以下顺序出现: {EXPECTED_HEADER}")

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第 {line_number} 行字段数量与表头不一致")

            sample_id = row["sample_id"].strip()
            name = row["name"].strip()
            if not sample_id:
                raise ValueError(f"第 {line_number} 行 sample_id 为空")
            if sample_id in seen_ids:
                raise ValueError(f"第 {line_number} 行 sample_id 重复: {sample_id}")
            if not name:
                raise ValueError(f"第 {line_number} 行 name 为空")

            try:
                features = [float(row["feature_1"]), float(row["feature_2"])]
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"第 {line_number} 行包含非法数值或标签") from exc
            if not np.all(np.isfinite(features)):
                raise ValueError(f"第 {line_number} 行特征包含 NaN 或无穷大")

            sample_ids.append(sample_id)
            names.append(name)
            feature_rows.append(features)
            labels.append(label)
            seen_ids.add(sample_id)

    if not feature_rows:
        raise ValueError("CSV 中没有数据行")

    return (
        sample_ids,
        names,
        np.asarray(feature_rows, dtype=float),
        np.asarray(labels, dtype=int),
    )


def compute_row_feature_means(X: np.ndarray) -> np.ndarray:
    """返回每个样本的特征均值。"""
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 必须只包含有限数值")
    return X.mean(axis=1)


def save_summary(
    path: Path,
    sample_ids: list[str],
    names: list[str],
    feature_means: np.ndarray,
    y: np.ndarray,
) -> None:
    """按固定表头和6位小数保存摘要。"""
    n_samples = len(sample_ids)
    if len(names) != n_samples or feature_means.shape != (n_samples,) or y.shape != (n_samples,):
        raise ValueError("所有输出字段的样本数必须一致")
    if len(set(sample_ids)) != n_samples or any(not value for value in sample_ids):
        raise ValueError("sample_ids 必须非空且唯一")
    if any(not value for value in names):
        raise ValueError("names 不能为空")
    if not np.all(np.isfinite(feature_means)):
        raise ValueError("feature_means 必须只包含有限数值")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", "name", "feature_mean", "label"])
        for sample_id, name, mean, label in zip(
            sample_ids, names, feature_means, y, strict=True
        ):
            writer.writerow([sample_id, name, f"{mean:.6f}", str(int(label))])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        sample_ids, names, X, y = load_mixed_dataset(args.input)
        feature_means = compute_row_feature_means(X)
        save_summary(args.output, sample_ids, names, feature_means, y)
    except (OSError, ValueError) as exc:
        print(f"CSV错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
