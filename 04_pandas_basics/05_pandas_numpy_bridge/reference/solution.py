"""参考实现：DataFrame与NumPy之间的形状、dtype和统计量。"""

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = ["sample_id", "prediction"]


def _validate_frame_and_features(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame必须是非空DataFrame")
    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or not all(isinstance(value, str) and value for value in feature_columns)
        or len(set(feature_columns)) != len(feature_columns)
    ):
        raise ValueError("feature_columns必须是不重复的非空字符串列表")
    missing = [value for value in feature_columns if value not in frame.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")


def _numeric_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    _validate_frame_and_features(frame, feature_columns)
    try:
        numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("特征列必须能完整转换为数值") from error
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("特征列不能包含缺失或非有限值")
    return numeric


def extract_supervised_arrays(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    *,
    integer_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = _numeric_feature_frame(frame, feature_columns)
    if not isinstance(label_column, str) or label_column not in frame.columns:
        raise ValueError("label_column必须是表中存在的列")
    if label_column in feature_columns:
        raise ValueError("标签列不能同时作为特征")
    try:
        labels = pd.to_numeric(frame.loc[:, label_column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("标签列必须能完整转换为数值") from error
    if labels.ndim != 1 or not np.all(np.isfinite(labels)):
        raise ValueError("标签必须是一维有限数值")
    if integer_labels:
        if not np.all(labels == np.floor(labels)):
            raise ValueError("整数标签模式不接受小数标签")
        labels = labels.astype(int)
    return numeric.to_numpy(dtype=float), labels


def fit_table_standardizer(
    train_frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.Series, pd.Series]:
    numeric = _numeric_feature_frame(train_frame, feature_columns)
    means = numeric.mean(axis=0).astype(float)
    scales = numeric.std(axis=0, ddof=0).astype(float)
    scales.loc[scales == 0] = 1.0
    means = means.reindex(feature_columns)
    scales = scales.reindex(feature_columns)
    return means, scales


def _validate_parameters(
    feature_columns: list[str], means: pd.Series, scales: pd.Series
) -> None:
    for name, values in (("means", means), ("scales", scales)):
        if not isinstance(values, pd.Series) or values.index.tolist() != feature_columns:
            raise ValueError(f"{name}索引必须与feature_columns顺序一致")
        if not np.all(np.isfinite(values.to_numpy(dtype=float))):
            raise ValueError(f"{name}必须包含有限数值")
    if np.any(scales.to_numpy(dtype=float) <= 0):
        raise ValueError("scales必须全部大于0")


def transform_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    scales: pd.Series,
) -> np.ndarray:
    numeric = _numeric_feature_frame(frame, feature_columns)
    _validate_parameters(feature_columns, means, scales)
    X = numeric.to_numpy(dtype=float)
    return (X - means.to_numpy(dtype=float)) / scales.to_numpy(dtype=float)


def prediction_frame(sample_ids: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    if not isinstance(sample_ids, pd.Series) or sample_ids.ndim != 1 or sample_ids.empty:
        raise ValueError("sample_ids必须是非空Series")
    if sample_ids.isna().any() or sample_ids.astype(str).str.strip().eq("").any():
        raise ValueError("sample_ids不能缺失或为空")
    if not isinstance(predictions, np.ndarray) or predictions.ndim != 1:
        raise ValueError("predictions必须具有形状(n,)，不能是(n,1)")
    if predictions.shape[0] != sample_ids.shape[0] or not np.all(np.isfinite(predictions)):
        raise ValueError("predictions必须与sample_ids等长且只含有限数值")
    return pd.DataFrame(
        {"sample_id": sample_ids.astype(str).to_numpy(), "prediction": predictions}
    )


def save_predictions(path: str | Path, frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError(f"预测表列必须按顺序为{PREDICTION_COLUMNS}")
    if frame.empty or frame.isna().any().any():
        raise ValueError("预测表不能为空或含缺失值")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        destination,
        index=False,
        columns=PREDICTION_COLUMNS,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
