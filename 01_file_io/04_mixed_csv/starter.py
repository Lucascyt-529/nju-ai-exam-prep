"""学生练习：标准库 csv 读取混合类型数据并输出摘要。"""

from pathlib import Path

import numpy as np


def load_mixed_dataset(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """返回 sample_ids、names、二维特征 X 和一维整数标签 y。"""
    raise NotImplementedError("请完成 load_mixed_dataset")


def compute_row_feature_means(X: np.ndarray) -> np.ndarray:
    """返回每个样本的特征均值。"""
    raise NotImplementedError("请完成 compute_row_feature_means")


def save_summary(
    path: Path,
    sample_ids: list[str],
    names: list[str],
    feature_means: np.ndarray,
    y: np.ndarray,
) -> None:
    """按固定表头和6位小数保存摘要。"""
    raise NotImplementedError("请完成 save_summary")


def main() -> int:
    """解析参数并组织完整流程。"""
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
