"""模拟卷1第1题参考实现。"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


FEATURES = ["feature_1", "feature_2", "feature_3"]
HEADER = ["sample_id", *FEATURES]


def load_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []; seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != HEADER:
            raise ValueError(f"表头必须为{HEADER}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"第{line_number}行字段数量错误")
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"第{line_number}行ID为空或重复")
            values = []
            for feature in FEATURES:
                text = row[feature].strip()
                if not text:
                    values.append(float("nan")); continue
                try:
                    value = float(text)
                except ValueError as exc:
                    raise ValueError(f"第{line_number}行包含非法数值") from exc
                if not np.isfinite(value):
                    raise ValueError(f"第{line_number}行数值必须有限或留空")
                values.append(value)
            rows.append(values); seen.add(sample_id)
    if not rows:
        raise ValueError("CSV不能为空")
    return np.asarray(rows, dtype=float)


def summarize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError("X形状必须为(n,3)")
    if np.any(np.all(np.isnan(X), axis=0)):
        raise ValueError("不能有整列缺失")
    means = np.nanmean(X, axis=0)
    filled = np.where(np.isnan(X), means[None, :], X)
    stds = np.std(filled, axis=0, ddof=0)
    missing = np.sum(np.isnan(X), axis=0)
    return means, stds, missing


def save_summary(path: Path, means: np.ndarray, stds: np.ndarray,
                 missing: np.ndarray) -> None:
    if means.shape != (3,) or stds.shape != (3,) or missing.shape != (3,):
        raise ValueError("统计量形状错误")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["feature", "mean", "std", "missing_count"])
        for name, mean, std, count in zip(FEATURES, means, stds, missing, strict=True):
            writer.writerow([name, f"{mean:.6f}", f"{std:.6f}", int(count)])


def run(input_path: Path, output_path: Path) -> None:
    save_summary(output_path, *summarize(load_csv(input_path)))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"第1题失败：{exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
