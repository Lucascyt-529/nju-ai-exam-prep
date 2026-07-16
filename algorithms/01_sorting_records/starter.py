"""学生练习：稳定归并排序完整程序。"""

from pathlib import Path


Record = tuple[str, int]


def parse_records(text: str) -> list[Record]:
    raise NotImplementedError("请完成 parse_records")


def stable_merge_sort_records(records: list[Record]) -> list[Record]:
    raise NotImplementedError("请完成 stable_merge_sort_records")


def format_records(records: list[Record]) -> str:
    raise NotImplementedError("请完成 format_records")


def solve(text: str) -> str:
    raise NotImplementedError("请完成 solve")


def run_cli(argv: list[str] | None = None) -> None:
    raise NotImplementedError("请完成 run_cli")
