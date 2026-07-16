"""学生练习：pandas Series 与 DataFrame。"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def make_score_series(
    names: Sequence[str], scores: Sequence[float]
) -> pd.Series:
    """创建以姓名为索引、名称为 score 的浮点 Series。"""
    raise NotImplementedError("请完成 make_score_series")


def make_student_frame(
    names: Sequence[str], study_hours: Sequence[float], scores: Sequence[float]
) -> pd.DataFrame:
    """创建列顺序固定为 name、study_hours、score 的 DataFrame。"""
    raise NotImplementedError("请完成 make_student_frame")


def describe_frame(frame: pd.DataFrame) -> dict[str, object]:
    """返回 shape、columns 和各列 dtype 字符串。"""
    raise NotImplementedError("请完成 describe_frame")


def select_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """按给定顺序选择多列并保持 DataFrame。"""
    raise NotImplementedError("请完成 select_columns")


def series_to_float_array(series: pd.Series) -> np.ndarray:
    """把 Series 转成独立的一维浮点 NumPy 数组。"""
    raise NotImplementedError("请完成 series_to_float_array")
