"""参考实现：使用 loadtxt 读取完整CSV并保存结果。"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 sample_ids、二维特征 X 和一维标签 y。"""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float, ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 3:
        raise ValueError("数据至少需要样本编号、一个特征和标签")
    if not np.all(np.isfinite(data)):
        raise ValueError("loadtxt 专题不允许缺失值或无穷大")
    if not np.all(data[:, 0] == np.floor(data[:, 0])):
        raise ValueError("sample_id 必须是整数")
    if not np.all(data[:, -1] == np.floor(data[:, -1])):
        raise ValueError("label 必须是整数")

    sample_ids = data[:, 0].astype(int)
    X = data[:, 1:-1]
    y = data[:, -1].astype(int)
    return sample_ids, X, y


def compute_feature_sums(X: np.ndarray) -> np.ndarray:
    """返回每个样本的特征和。"""
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    return X.sum(axis=1)


def save_results(path: Path, sample_ids: np.ndarray, values: np.ndarray) -> None:
    """保存 sample_id 和6位小数结果。"""
    if sample_ids.ndim != 1 or values.ndim != 1:
        raise ValueError("sample_ids 和 values 必须是一维数组")
    if sample_ids.shape[0] != values.shape[0]:
        raise ValueError("sample_ids 和 values 长度必须一致")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.column_stack((sample_ids, values))
    with path.open("w", encoding="utf-8", newline="\n") as file:
        np.savetxt(
            file,
            output,
            delimiter=",",
            header="sample_id,feature_sum",
            comments="",
            fmt=["%d", "%.6f"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        sample_ids, X, _ = load_dataset(args.input)
        values = compute_feature_sums(X)
        save_results(args.output, sample_ids, values)
    except (OSError, ValueError) as exc:
        print(f"CSV错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
