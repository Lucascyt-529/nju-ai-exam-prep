"""参考实现：pandas 缺失值、重复值和类型转换。"""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = ["student_id", "name", "study_hours", "attendance"]
NUMERIC_COLUMNS = ["study_hours", "attendance"]


def _validate_columns(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame 必须是 DataFrame")
    if frame.columns.tolist() != COLUMNS:
        raise ValueError(f"列必须按以下顺序出现: {COLUMNS}")
    if frame.empty:
        raise ValueError("数据表不能为空")


def read_raw_table(path: str | Path) -> pd.DataFrame:
    """把所有列按原始字符串读取，空字段暂时保留为空字符串。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到数据文件: {source}")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    _validate_columns(frame)
    return frame


def normalize_and_convert(frame: pd.DataFrame) -> pd.DataFrame:
    """清理字符串并把两个数值列转换为浮点数，非法值变为 NaN。"""
    _validate_columns(frame)
    cleaned = frame.copy()
    for column in ["student_id", "name"]:
        cleaned[column] = cleaned[column].astype(str).str.strip()
    if (cleaned["student_id"] == "").any() or (cleaned["name"] == "").any():
        raise ValueError("student_id 和 name 不能为空")

    for column in NUMERIC_COLUMNS:
        text = cleaned[column].astype(str).str.strip()
        cleaned[column] = pd.to_numeric(text, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return cleaned


def drop_duplicate_students(frame: pd.DataFrame) -> pd.DataFrame:
    """按 student_id 保留最后一条记录并重置行索引。"""
    _validate_columns(frame)
    if frame["student_id"].isna().any() or (frame["student_id"] == "").any():
        raise ValueError("student_id 不能为空")
    return frame.drop_duplicates("student_id", keep="last").reset_index(drop=True)


def fit_numeric_medians(frame: pd.DataFrame) -> pd.Series:
    """只从训练表拟合 study_hours 和 attendance 的中位数。"""
    _validate_columns(frame)
    medians = frame[NUMERIC_COLUMNS].median()
    if medians.isna().any() or not np.all(np.isfinite(medians.to_numpy())):
        raise ValueError("存在无法拟合中位数的数值列")
    return medians.astype(float)


def fill_numeric_missing(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """使用已有中位数填补副本，不重新拟合。"""
    _validate_columns(frame)
    if not isinstance(medians, pd.Series) or medians.index.tolist() != NUMERIC_COLUMNS:
        raise ValueError(f"medians 索引必须是: {NUMERIC_COLUMNS}")
    if medians.isna().any() or not np.all(np.isfinite(medians.to_numpy(dtype=float))):
        raise ValueError("medians 必须只包含有限数值")

    filled = frame.copy()
    filled[NUMERIC_COLUMNS] = filled[NUMERIC_COLUMNS].fillna(medians)
    values = filled[NUMERIC_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("填补后仍存在非有限数值")
    return filled


def save_cleaned_table(path: str | Path, frame: pd.DataFrame) -> None:
    """按固定列顺序和6位小数保存清洗结果。"""
    _validate_columns(frame)
    if frame.isna().any().any():
        raise ValueError("保存前不能存在缺失值")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        destination,
        index=False,
        columns=COLUMNS,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
