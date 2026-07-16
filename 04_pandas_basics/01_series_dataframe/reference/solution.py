"""参考实现：pandas Series 与 DataFrame。"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def make_score_series(
    names: Sequence[str], scores: Sequence[float]
) -> pd.Series:
    """创建以姓名为索引、名称为 score 的浮点 Series。"""
    if len(names) == 0 or len(names) != len(scores):
        raise ValueError("names 和 scores 必须非空且长度一致")
    if len(set(names)) != len(names):
        raise ValueError("姓名索引不能重复")
    try:
        result = pd.Series(scores, index=list(names), name="score", dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores 必须是数值序列") from exc
    if not np.all(np.isfinite(result.to_numpy())):
        raise ValueError("scores 必须只包含有限数值")
    return result


def make_student_frame(
    names: Sequence[str], study_hours: Sequence[float], scores: Sequence[float]
) -> pd.DataFrame:
    """创建列顺序固定为 name、study_hours、score 的 DataFrame。"""
    lengths = {len(names), len(study_hours), len(scores)}
    if len(lengths) != 1 or len(names) == 0:
        raise ValueError("三列必须非空且长度一致")
    try:
        frame = pd.DataFrame(
            {
                "name": list(names),
                "study_hours": pd.Series(study_hours, dtype=float),
                "score": pd.Series(scores, dtype=float),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("学习时长和分数必须是数值") from exc
    if frame["name"].isna().any() or (frame["name"].astype(str).str.len() == 0).any():
        raise ValueError("姓名不能为空")
    if not np.all(np.isfinite(frame[["study_hours", "score"]].to_numpy())):
        raise ValueError("数值列必须只包含有限值")
    return frame


def describe_frame(frame: pd.DataFrame) -> dict[str, object]:
    """返回 shape、columns 和各列 dtype 字符串。"""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 DataFrame")
    return {
        "shape": frame.shape,
        "columns": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }


def select_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """按给定顺序选择多列并保持 DataFrame。"""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 DataFrame")
    requested = list(columns)
    if not requested:
        raise ValueError("至少选择一列")
    missing = [column for column in requested if column not in frame.columns]
    if missing:
        raise KeyError(f"不存在的列: {missing}")
    return frame.loc[:, requested].copy()


def series_to_float_array(series: pd.Series) -> np.ndarray:
    """把 Series 转成独立的一维浮点 NumPy 数组。"""
    if not isinstance(series, pd.Series):
        raise TypeError("series 必须是 Series")
    try:
        result = series.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Series 必须能够转换为浮点数组") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("转换结果必须只包含有限数值")
    return result
