"""学生练习：使用 genfromtxt 读取缺失值CSV并保存清洗结果。"""

from pathlib import Path

import numpy as np


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 sample_ids、可能含NaN的二维特征 X 和一维标签 y。"""
    raise NotImplementedError("请完成 load_dataset")


def fit_and_fill_feature_means(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算特征均值并返回填补后的副本和均值。"""
    raise NotImplementedError("请完成 fit_and_fill_feature_means")


def save_cleaned_dataset(
    path: Path, sample_ids: np.ndarray, X: np.ndarray, y: np.ndarray
) -> None:
    """保存清洗后的完整CSV。"""
    raise NotImplementedError("请完成 save_cleaned_dataset")


def main() -> int:
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
