"""学生练习：把纯函数串成可测试的命令行完整程序。"""

import argparse
from pathlib import Path

import numpy as np


def load_values(path: Path) -> np.ndarray:
    raise NotImplementedError("第1步：完成 load_values")


def compute_statistics(values: np.ndarray, *, scale: float = 1.0) -> dict[str, float | int]:
    raise NotImplementedError("第2步：完成 compute_statistics")


def save_report(path: Path, report: dict[str, float | int]) -> None:
    raise NotImplementedError("第3步：完成 save_report")


def run_pipeline(input_path: Path, output_path: Path, *, scale: float = 1.0) -> None:
    raise NotImplementedError("第4步：串联读取、计算和保存")


def build_parser() -> argparse.ArgumentParser:
    raise NotImplementedError("第5步：声明命令行参数")


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("第6步：解析参数、捕获预期错误并返回错误码")


if __name__ == "__main__":
    raise SystemExit(main())
