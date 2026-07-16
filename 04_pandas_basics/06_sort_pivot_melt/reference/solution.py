"""参考实现：稳定排序、pivot、pivot_table与melt。"""

from pathlib import Path

import pandas as pd


ALLOWED_AGGREGATIONS = {"mean", "sum", "max", "min"}


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame必须是非空DataFrame")


def _validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    _validate_frame(frame)
    if any(not isinstance(value, str) or value not in frame.columns for value in columns):
        raise ValueError("指定列必须存在于DataFrame")


def stable_sort_records(
    frame: pd.DataFrame,
    by: list[str],
    ascending: list[bool],
    *,
    reset_index: bool = True,
) -> pd.DataFrame:
    _validate_frame(frame)
    if (
        not isinstance(by, list)
        or not by
        or len(set(by)) != len(by)
        or not isinstance(ascending, list)
        or len(ascending) != len(by)
        or not all(isinstance(value, bool) for value in ascending)
    ):
        raise ValueError("by与ascending必须是等长有效列表")
    _validate_columns(frame, by)
    result = frame.sort_values(by=by, ascending=ascending, kind="stable")
    return result.reset_index(drop=True) if reset_index else result.copy()


def _finished_wide(frame: pd.DataFrame, index: str) -> pd.DataFrame:
    result = frame.sort_index(axis=0).sort_index(axis=1).reset_index()
    result.columns.name = None
    ordered = [index] + sorted(value for value in result.columns if value != index)
    return result.loc[:, ordered].reset_index(drop=True)


def strict_long_to_wide(
    frame: pd.DataFrame, *, index: str, columns: str, values: str
) -> pd.DataFrame:
    _validate_columns(frame, [index, columns, values])
    duplicate = frame.duplicated([index, columns], keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, [index, columns]].drop_duplicates().to_dict("records")
        raise ValueError(f"pivot键不唯一: {keys}")
    wide = frame.pivot(index=index, columns=columns, values=values)
    return _finished_wide(wide, index)


def aggregate_long_to_wide(
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    aggfunc: str,
) -> pd.DataFrame:
    _validate_columns(frame, [index, columns, values])
    if aggfunc not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"aggfunc必须属于{sorted(ALLOWED_AGGREGATIONS)}")
    wide = frame.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        sort=True,
        observed=False,
    )
    return _finished_wide(wide, index)


def wide_to_long(
    frame: pd.DataFrame,
    *,
    index: str,
    variable_name: str,
    value_name: str,
    drop_missing: bool = True,
) -> pd.DataFrame:
    _validate_columns(frame, [index])
    if not isinstance(variable_name, str) or not variable_name or not isinstance(value_name, str) or not value_name:
        raise ValueError("variable_name和value_name必须是非空字符串")
    if variable_name == value_name or variable_name in frame.columns or value_name in frame.columns:
        raise ValueError("新增长表列名不能冲突")
    value_columns = [value for value in frame.columns if value != index]
    if not value_columns:
        raise ValueError("宽表必须至少包含一个值列")
    result = frame.melt(
        id_vars=[index],
        value_vars=value_columns,
        var_name=variable_name,
        value_name=value_name,
    )
    if drop_missing:
        result = result.dropna(subset=[value_name])
    return result.sort_values([index, variable_name], kind="stable").reset_index(drop=True)


def save_wide_table(path: str | Path, frame: pd.DataFrame) -> None:
    _validate_frame(frame)
    if frame.columns.name is not None:
        raise ValueError("保存前必须清除columns.name")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        destination,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
