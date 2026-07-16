"""学生练习：使用 loadtxt 读取完整CSV并保存结果。"""

from pathlib import Path

import numpy as np


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 sample_ids、二维特征 X 和一维标签 y。"""
    raise NotImplementedError("请完成 load_dataset")


def compute_feature_sums(X: np.ndarray) -> np.ndarray:
    """返回每个样本的特征和。"""
    raise NotImplementedError("请完成 compute_feature_sums")


def save_results(path: Path, sample_ids: np.ndarray, values: np.ndarray) -> None:
    """保存 sample_id 和6位小数结果。"""
    raise NotImplementedError("请完成 save_results")


def main() -> int:
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
