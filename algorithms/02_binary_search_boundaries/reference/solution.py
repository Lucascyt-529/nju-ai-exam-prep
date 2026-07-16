"""参考实现：二分边界完整程序。"""

import argparse
from pathlib import Path
import sys


def lower_bound(values: list[int], target: int) -> int:
    left = 0
    right = len(values)
    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


def upper_bound(values: list[int], target: int) -> int:
    left = 0
    right = len(values)
    while left < right:
        middle = (left + right) // 2
        if values[middle] <= target:
            left = middle + 1
        else:
            right = middle
    return left


def _parse_int_line(line: str, expected: int, name: str) -> list[int]:
    parts = line.split()
    if len(parts) != expected:
        raise ValueError(f"{name}数量应为{expected}，实际为{len(parts)}")
    try:
        return [int(value) for value in parts]
    except ValueError as error:
        raise ValueError(f"{name}必须全部是整数") from error


def parse_problem(text: str) -> tuple[list[int], list[int]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    lines = text.strip().splitlines()
    if len(lines) != 3:
        raise ValueError("输入必须恰好包含三行")
    first = _parse_int_line(lines[0], 2, "n和q")
    n, q = first
    if n < 0 or q <= 0:
        raise ValueError("n必须非负且q必须为正")
    values = _parse_int_line(lines[1], n, "数组元素")
    queries = _parse_int_line(lines[2], q, "查询")
    if any(values[index] > values[index + 1] for index in range(n - 1)):
        raise ValueError("输入数组必须非降序")
    return values, queries


def solve(text: str) -> str:
    values, queries = parse_problem(text)
    lines = []
    for target in queries:
        lower = lower_bound(values, target)
        upper = upper_bound(values, target)
        lines.append(f"{target} {lower} {upper} {upper - lower}")
    return "\n".join(lines)


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="二分查找上下边界")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("output_path", nargs="?")
    args = parser.parse_args(argv)
    input_text = (
        Path(args.input_path).read_text(encoding="utf-8")
        if args.input_path
        else sys.stdin.read()
    )
    output_text = solve(input_text) + "\n"
    if args.output_path:
        destination = Path(args.output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output_text, encoding="utf-8", newline="")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    run_cli()
