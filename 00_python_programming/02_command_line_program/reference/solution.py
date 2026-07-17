"""参考实现：可测试、可复写的命令行完整程序骨架。"""

import argparse
from numbers import Real
from pathlib import Path
import sys

import numpy as np


def load_values(path: Path) -> np.ndarray:
    values: list[float] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 1:
            raise ValueError(f"第{line_number}行必须恰好包含一个数")
        try:
            value = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"第{line_number}行不是合法数值") from exc
        if not np.isfinite(value):
            raise ValueError(f"第{line_number}行必须是有限数值")
        values.append(value)
    if not values:
        raise ValueError("输入中没有有效数据行")
    return np.asarray(values, dtype=float)


def compute_statistics(values: np.ndarray, *, scale: float = 1.0) -> dict[str, float | int]:
    if (not isinstance(values, np.ndarray) or values.ndim != 1 or values.size == 0
            or not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values))):
        raise ValueError("values必须是非空有限数值一维数组")
    if (isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real)
            or not np.isfinite(scale)):
        raise ValueError("scale必须是有限实数")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "scaled_mean": float(np.mean(values) * float(scale)),
    }


def save_report(path: Path, report: dict[str, float | int]) -> None:
    expected = ["count", "mean", "std", "minimum", "maximum", "scaled_mean"]
    if list(report) != expected:
        raise ValueError(f"report必须按顺序包含: {expected}")
    if (isinstance(report["count"], bool) or not isinstance(report["count"], int)
            or report["count"] <= 0
            or any(not np.isfinite(report[key]) for key in expected[1:])):
        raise ValueError("report数值无效")
    lines = [f"count={report['count']}"]
    lines.extend(f"{key}={float(report[key]):.6f}" for key in expected[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_pipeline(input_path: Path, output_path: Path, *, scale: float = 1.0) -> None:
    values = load_values(input_path)
    report = compute_statistics(values, scale=scale)
    save_report(output_path, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8输入文件")
    parser.add_argument("--output", type=Path, required=True, help="报告输出路径")
    parser.add_argument("--scale", type=float, default=1.0, help="仅用于scaled_mean")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_pipeline(args.input, args.output, scale=args.scale)
    except (OSError, ValueError) as exc:
        print(f"程序失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
