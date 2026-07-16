"""参考实现：使用 genfromtxt 读取缺失值CSV并保存清洗结果。"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 sample_ids、可能含NaN的二维特征 X 和一维标签 y。"""
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        dtype=float,
        missing_values="",
        filling_values=np.nan,
        ndmin=2,
    )
    if data.shape[0] == 0 or data.shape[1] < 3:
        raise ValueError("数据至少需要样本编号、一个特征和标签")
    if np.any(np.isinf(data)):
        raise ValueError("数据不能包含无穷大")
    if np.any(np.isnan(data[:, 0])) or np.any(np.isnan(data[:, -1])):
        raise ValueError("sample_id 和 label 不允许缺失")
    if not np.all(data[:, 0] == np.floor(data[:, 0])):
        raise ValueError("sample_id 必须是整数")
    if not np.all(data[:, -1] == np.floor(data[:, -1])):
        raise ValueError("label 必须是整数")

    return data[:, 0].astype(int), data[:, 1:-1], data[:, -1].astype(int)


def fit_and_fill_feature_means(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算特征均值并返回填补后的副本和均值。"""
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X 必须是非空二维数组")
    all_missing = np.all(np.isnan(X), axis=0)
    if np.any(all_missing):
        raise ValueError(f"存在整列缺失: {np.flatnonzero(all_missing).tolist()}")
    means = np.nanmean(X, axis=0)
    filled = np.where(np.isnan(X), means, X)
    return filled, means


def save_cleaned_dataset(
    path: Path, sample_ids: np.ndarray, X: np.ndarray, y: np.ndarray
) -> None:
    """保存清洗后的完整CSV。"""
    if X.ndim != 2 or sample_ids.ndim != 1 or y.ndim != 1:
        raise ValueError("sample_ids和y必须一维，X必须二维")
    if not (sample_ids.shape[0] == X.shape[0] == y.shape[0]):
        raise ValueError("sample_ids、X和y的样本数必须一致")
    if X.shape[1] != 2:
        raise ValueError("本练习输出格式固定为两个特征")
    if not np.all(np.isfinite(X)):
        raise ValueError("保存前的X必须全部为有限数值")

    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.column_stack((sample_ids, X, y))
    with path.open("w", encoding="utf-8", newline="\n") as file:
        np.savetxt(
            file,
            output,
            delimiter=",",
            header="sample_id,feature_1,feature_2,label",
            comments="",
            fmt=["%d", "%.6f", "%.6f", "%d"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        sample_ids, X, y = load_dataset(args.input)
        filled, _ = fit_and_fill_feature_means(X)
        save_cleaned_dataset(args.output, sample_ids, filled, y)
    except (OSError, ValueError) as exc:
        print(f"CSV错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
