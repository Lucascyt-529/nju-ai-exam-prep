"""参考实现：稳定归并排序完整程序。"""

import argparse
from pathlib import Path
import sys


Record = tuple[str, int]


def parse_records(text: str) -> list[Record]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    lines = text.strip().splitlines()
    if len(lines[0].split()) != 1:
        raise ValueError("第一行必须只包含记录数n")
    try:
        n = int(lines[0])
    except ValueError as error:
        raise ValueError("n必须是整数") from error
    if n < 0 or len(lines) != n + 1:
        raise ValueError("记录数与实际行数不一致")
    records: list[Record] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"第{line_number}行必须是: student_id score")
        student_id, score_text = parts
        if student_id in seen_ids:
            raise ValueError(f"student_id重复: {student_id}")
        try:
            score = int(score_text)
        except ValueError as error:
            raise ValueError(f"第{line_number}行score必须是整数") from error
        seen_ids.add(student_id)
        records.append((student_id, score))
    return records


def _merge(left: list[Record], right: list[Record]) -> list[Record]:
    result: list[Record] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index][1] >= right[right_index][1]:
            result.append(left[left_index])
            left_index += 1
        else:
            result.append(right[right_index])
            right_index += 1
    result.extend(left[left_index:])
    result.extend(right[right_index:])
    return result


def stable_merge_sort_records(records: list[Record]) -> list[Record]:
    if not isinstance(records, list) or any(
        not isinstance(record, tuple)
        or len(record) != 2
        or not isinstance(record[0], str)
        or not isinstance(record[1], int)
        for record in records
    ):
        raise TypeError("records必须是(str, int)元组列表")
    if len(records) <= 1:
        return records.copy()
    middle = len(records) // 2
    left = stable_merge_sort_records(records[:middle])
    right = stable_merge_sort_records(records[middle:])
    return _merge(left, right)


def format_records(records: list[Record]) -> str:
    return "\n".join(f"{student_id} {score}" for student_id, score in records)


def solve(text: str) -> str:
    return format_records(stable_merge_sort_records(parse_records(text)))


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="稳定归并排序学生记录")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("output_path", nargs="?")
    args = parser.parse_args(argv)
    input_text = (
        Path(args.input_path).read_text(encoding="utf-8")
        if args.input_path
        else sys.stdin.read()
    )
    result = solve(input_text)
    output_text = result + ("\n" if result else "")
    if args.output_path:
        destination = Path(args.output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output_text, encoding="utf-8", newline="")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    run_cli()
