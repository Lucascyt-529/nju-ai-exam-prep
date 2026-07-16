"""学生练习：pandas 缺失值、重复值和类型转换。"""

from pathlib import Path

import pandas as pd


def read_raw_table(path: str | Path) -> pd.DataFrame:
    """把所有列按原始字符串读取，空字段暂时保留为空字符串。"""
    raise NotImplementedError("请完成 read_raw_table")


def normalize_and_convert(frame: pd.DataFrame) -> pd.DataFrame:
    """清理字符串并把两个数值列转换为浮点数，非法值变为 NaN。"""
    raise NotImplementedError("请完成 normalize_and_convert")


def drop_duplicate_students(frame: pd.DataFrame) -> pd.DataFrame:
    """按 student_id 保留最后一条记录并重置行索引。"""
    raise NotImplementedError("请完成 drop_duplicate_students")


def fit_numeric_medians(frame: pd.DataFrame) -> pd.Series:
    """只从训练表拟合 study_hours 和 attendance 的中位数。"""
    raise NotImplementedError("请完成 fit_numeric_medians")


def fill_numeric_missing(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """使用已有中位数填补副本，不重新拟合。"""
    raise NotImplementedError("请完成 fill_numeric_missing")


def save_cleaned_table(path: str | Path, frame: pd.DataFrame) -> None:
    """按固定列顺序和6位小数保存清洗结果。"""
    raise NotImplementedError("请完成 save_cleaned_table")
