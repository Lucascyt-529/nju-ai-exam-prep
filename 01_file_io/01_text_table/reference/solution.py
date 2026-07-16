"""参考实现：读取TXT数值表并保存列均值。"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_numeric_table(path: Path) -> np.ndarray:
    """读取允许空行但列数必须一致的数值表。"""
    rows: list[list[float]] = []
    expected_columns: int | None = None

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if expected_columns is None:
            expected_columns = len(fields)
        elif len(fields) != expected_columns:
            raise ValueError(
                f"第 {line_number} 行应有 {expected_columns} 列，实际得到 {len(fields)} 列"
            )

        try:
            rows.append([float(value) for value in fields])
        except ValueError as exc:
            raise ValueError(f"第 {line_number} 行包含非数值字段") from exc

    if not rows:
        raise ValueError("文件中没有有效数据")

    table = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(table)):
        raise ValueError("数据中包含 NaN 或无穷大")
    return table


def column_means(table: np.ndarray) -> np.ndarray:
    """计算每一列的平均值。"""
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] == 0:
        raise ValueError("table 必须是非空二维数组")
    return table.mean(axis=0)


def save_column_means(path: Path, means: np.ndarray) -> None:
    """按6位小数将结果写入一行文本。"""
    if means.ndim != 1:
        raise ValueError("means 必须是一维数组")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = " ".join(f"{value:.6f}" for value in means) + "\n"
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="输入TXT文件")
    parser.add_argument("--output", type=Path, required=True, help="输出结果文件")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        table = load_numeric_table(args.input)
        means = column_means(table)
        save_column_means(args.output, means)
    except (OSError, ValueError) as exc:
        print(f"数据错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
