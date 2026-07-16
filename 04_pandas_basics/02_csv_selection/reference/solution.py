"""参考实现：pandas 读取 CSV 与选择数据。"""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "student_id",
    "name",
    "study_hours",
    "attendance",
    "score",
]


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 DataFrame")
    if frame.columns.tolist() != REQUIRED_COLUMNS:
        raise ValueError(f"列必须按以下顺序出现: {REQUIRED_COLUMNS}")
    if frame.empty:
        raise ValueError("数据表不能为空")


def load_students(path: str | Path) -> pd.DataFrame:
    """读取学生 CSV，保留 student_id 为字符串，并检查列顺序。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到数据文件: {source}")
    frame = pd.read_csv(source, dtype={"student_id": str})
    _validate_frame(frame)
    if frame.isna().any().any():
        raise ValueError("本专题数据暂不允许缺失值")
    numeric = frame[["study_hours", "attendance", "score"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("数值列必须只包含有限值")
    return frame


def select_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """按顺序返回 study_hours 和 attendance 两列。"""
    _validate_frame(frame)
    return frame.loc[:, ["study_hours", "attendance"]].copy()


def select_rows_by_position(
    frame: pd.DataFrame, start: int, stop: int
) -> pd.DataFrame:
    """使用左闭右开的位置切片选择行。"""
    _validate_frame(frame)
    if not isinstance(start, int) or not isinstance(stop, int):
        raise TypeError("start 和 stop 必须是整数")
    return frame.iloc[start:stop].copy()


def filter_by_score(frame: pd.DataFrame, minimum: float) -> pd.DataFrame:
    """保留 score 大于等于 minimum 的行，不重置原索引。"""
    _validate_frame(frame)
    if not np.isfinite(minimum):
        raise ValueError("minimum 必须是有限数值")
    return frame.loc[frame["score"] >= minimum].copy()
