"""学生练习：稳定排序、pivot、pivot_table与melt。"""

from pathlib import Path

import pandas as pd


def stable_sort_records(
    frame: pd.DataFrame,
    by: list[str],
    ascending: list[bool],
    *,
    reset_index: bool = True,
) -> pd.DataFrame:
    raise NotImplementedError("请完成 stable_sort_records")


def strict_long_to_wide(
    frame: pd.DataFrame, *, index: str, columns: str, values: str
) -> pd.DataFrame:
    raise NotImplementedError("请完成 strict_long_to_wide")


def aggregate_long_to_wide(
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    aggfunc: str,
) -> pd.DataFrame:
    raise NotImplementedError("请完成 aggregate_long_to_wide")


def wide_to_long(
    frame: pd.DataFrame,
    *,
    index: str,
    variable_name: str,
    value_name: str,
    drop_missing: bool = True,
) -> pd.DataFrame:
    raise NotImplementedError("请完成 wide_to_long")


def save_wide_table(path: str | Path, frame: pd.DataFrame) -> None:
    raise NotImplementedError("请完成 save_wide_table")
