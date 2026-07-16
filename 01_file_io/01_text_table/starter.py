"""学生练习：读取TXT数值表并保存列均值。"""

from pathlib import Path

import numpy as np


def load_numeric_table(path: Path) -> np.ndarray:
    """读取允许空行但列数必须一致的数值表。"""
    raise NotImplementedError("请完成 load_numeric_table")


def column_means(table: np.ndarray) -> np.ndarray:
    """计算每一列的平均值。"""
    raise NotImplementedError("请完成 column_means")


def save_column_means(path: Path, means: np.ndarray) -> None:
    """按6位小数将结果写入一行文本。"""
    raise NotImplementedError("请完成 save_column_means")


def main() -> int:
    """解析命令行参数并组织完整文件处理流程。"""
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
