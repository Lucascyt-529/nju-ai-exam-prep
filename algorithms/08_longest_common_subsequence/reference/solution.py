"""参考实现：二维DP最长公共子序列。"""

import argparse
from pathlib import Path
import sys


def parse_sequences(text: str) -> tuple[str, str]:
    if not isinstance(text, str):
        raise ValueError("输入必须是字符串")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != 2:
        raise ValueError("输入必须恰好包含两个序列行")
    if any(any(character.isspace() for character in line) for line in lines):
        raise ValueError("序列内部不能包含空白字符")
    return lines[0], lines[1]


def lcs_table(first: str, second: str) -> list[list[int]]:
    if not isinstance(first, str) or not isinstance(second, str):
        raise TypeError("first和second必须是字符串")
    table = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for first_index in range(1, len(first) + 1):
        for second_index in range(1, len(second) + 1):
            if first[first_index - 1] == second[second_index - 1]:
                table[first_index][second_index] = table[first_index - 1][second_index - 1] + 1
            else:
                table[first_index][second_index] = max(
                    table[first_index - 1][second_index], table[first_index][second_index - 1]
                )
    return table


def reconstruct_lcs(first: str, second: str, table: list[list[int]]) -> str:
    if len(table) != len(first) + 1 or any(len(row) != len(second) + 1 for row in table):
        raise ValueError("table形状与两个序列不匹配")
    first_index = len(first)
    second_index = len(second)
    reversed_characters = []
    while first_index > 0 and second_index > 0:
        if first[first_index - 1] == second[second_index - 1]:
            reversed_characters.append(first[first_index - 1])
            first_index -= 1
            second_index -= 1
        elif table[first_index - 1][second_index] >= table[first_index][second_index - 1]:
            first_index -= 1
        else:
            second_index -= 1
    return "".join(reversed(reversed_characters))


def solve(text: str) -> str:
    first, second = parse_sequences(text)
    table = lcs_table(first, second)
    sequence = reconstruct_lcs(first, second, table)
    return f"length: {table[-1][-1]}\nsequence: {sequence}"


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="二维DP最长公共子序列")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("output_path", nargs="?")
    args = parser.parse_args(argv)
    text = Path(args.input_path).read_text(encoding="utf-8") if args.input_path else sys.stdin.read()
    output = solve(text) + "\n"
    if args.output_path:
        destination = Path(args.output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8", newline="")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    run_cli()
