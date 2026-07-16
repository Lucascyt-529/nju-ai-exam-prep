"""参考实现：哈希计数、保序去重与两数配对完整程序。"""

import argparse
from pathlib import Path
import sys


def _validate_values(values: list[int]) -> None:
    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise TypeError("values必须是整数列表")


def parse_problem(text: str) -> tuple[list[int], int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != 2:
        raise ValueError("输入必须恰好包含两行")
    first = lines[0].split()
    if len(first) != 2:
        raise ValueError("第一行必须是n和target")
    try:
        n, target = map(int, first)
    except ValueError as error:
        raise ValueError("n和target必须是整数") from error
    if n < 0:
        raise ValueError("n必须非负")
    parts = lines[1].split()
    if len(parts) != n:
        raise ValueError(f"序列长度应为{n}，实际为{len(parts)}")
    try:
        values = [int(value) for value in parts]
    except ValueError as error:
        raise ValueError("序列必须全部是整数") from error
    return values, target


def frequency_in_first_order(values: list[int]) -> list[tuple[int, int]]:
    _validate_values(values)
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in values:
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    return [(value, counts[value]) for value in order]


def stable_unique(values: list[int]) -> list[int]:
    return [value for value, _ in frequency_in_first_order(values)]


def first_two_sum_pair(values: list[int], target: int) -> tuple[int, int] | None:
    _validate_values(values)
    if not isinstance(target, int) or isinstance(target, bool):
        raise TypeError("target必须是整数")
    earliest_index: dict[int, int] = {}
    for right_index, value in enumerate(values):
        complement = target - value
        if complement in earliest_index:
            return earliest_index[complement], right_index
        earliest_index.setdefault(value, right_index)
    return None


def solve(text: str) -> str:
    values, target = parse_problem(text)
    frequencies = frequency_in_first_order(values)
    unique_text = " ".join(str(value) for value, _ in frequencies)
    frequency_text = " ".join(f"{value}:{count}" for value, count in frequencies)
    pair = first_two_sum_pair(values, target)
    pair_text = "NONE" if pair is None else f"{pair[0]} {pair[1]}"
    return "\n".join(
        [
            f"unique:{' ' + unique_text if unique_text else ''}",
            f"frequencies:{' ' + frequency_text if frequency_text else ''}",
            f"pair: {pair_text}",
        ]
    )


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="哈希计数、保序去重与两数配对")
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
