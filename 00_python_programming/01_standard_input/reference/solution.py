"""参考实现：从标准输入读取矩阵并输出列均值。"""

import sys
from typing import TextIO

import numpy as np


def parse_matrix(stream: TextIO) -> np.ndarray:
    """读取 n、d 和矩阵，返回形状为 (n, d) 的浮点数组。"""
    lines = [line.strip() for line in stream if line.strip()]
    if not lines:
        raise ValueError("输入为空")

    header = lines[0].split()
    if len(header) != 2:
        raise ValueError("第一行必须恰好包含 n 和 d")

    try:
        n, d = (int(value) for value in header)
    except ValueError as exc:
        raise ValueError("n 和 d 必须是整数") from exc

    if n <= 0 or d <= 0:
        raise ValueError("n 和 d 必须是正整数")

    data_lines = lines[1:]
    if len(data_lines) != n:
        raise ValueError(f"应有 {n} 行数据，实际得到 {len(data_lines)} 行")

    rows: list[list[float]] = []
    for line_number, line in enumerate(data_lines, start=2):
        fields = line.split()
        if len(fields) != d:
            raise ValueError(
                f"第 {line_number} 行应有 {d} 列，实际得到 {len(fields)} 列"
            )
        try:
            rows.append([float(value) for value in fields])
        except ValueError as exc:
            raise ValueError(f"第 {line_number} 行包含非数值字段") from exc

    matrix = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("矩阵中包含 NaN 或无穷大")

    return matrix


def column_means(matrix: np.ndarray) -> np.ndarray:
    """返回每一列的平均值。"""
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix 必须是非空二维数组")
    return matrix.mean(axis=0)


def format_result(means: np.ndarray) -> str:
    """将列均值格式化为保留6位小数的一行文本。"""
    if means.ndim != 1:
        raise ValueError("means 必须是一维数组")
    return " ".join(f"{value:.6f}" for value in means)


def main() -> int:
    """组织读取、计算、输出和错误处理。"""
    try:
        matrix = parse_matrix(sys.stdin)
        means = column_means(matrix)
        print(format_result(means))
    except ValueError as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
