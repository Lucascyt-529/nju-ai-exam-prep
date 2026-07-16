"""学生练习：从标准输入读取矩阵并输出列均值。"""

import sys
from typing import TextIO

import numpy as np


def parse_matrix(stream: TextIO) -> np.ndarray:
    """读取 n、d 和矩阵，返回形状为 (n, d) 的浮点数组。"""
    raise NotImplementedError("请完成 parse_matrix")


def column_means(matrix: np.ndarray) -> np.ndarray:
    """返回每一列的平均值。"""
    raise NotImplementedError("请完成 column_means")


def format_result(means: np.ndarray) -> str:
    """将列均值格式化为保留6位小数的一行文本。"""
    raise NotImplementedError("请完成 format_result")


def main() -> int:
    """组织读取、计算、输出和错误处理。"""
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
