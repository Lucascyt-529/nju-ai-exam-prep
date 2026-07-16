"""学生练习：pandas 读取 CSV 与选择数据。"""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "student_id",
    "name",
    "study_hours",
    "attendance",
    "score",
]


def load_students(path: str | Path) -> pd.DataFrame:
    """读取学生 CSV，保留 student_id 为字符串，并检查列顺序。"""
    raise NotImplementedError("请完成 load_students")


def select_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """按顺序返回 study_hours 和 attendance 两列。"""
    raise NotImplementedError("请完成 select_feature_columns")


def select_rows_by_position(
    frame: pd.DataFrame, start: int, stop: int
) -> pd.DataFrame:
    """使用左闭右开的位置切片选择行。"""
    raise NotImplementedError("请完成 select_rows_by_position")


def filter_by_score(frame: pd.DataFrame, minimum: float) -> pd.DataFrame:
    """保留 score 大于等于 minimum 的行，不重置原索引。"""
    raise NotImplementedError("请完成 filter_by_score")
