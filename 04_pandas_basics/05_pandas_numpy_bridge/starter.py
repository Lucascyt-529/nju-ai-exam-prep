"""学生练习：DataFrame与NumPy之间的形状、dtype和统计量。"""

from pathlib import Path

import numpy as np
import pandas as pd


def extract_supervised_arrays(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    *,
    integer_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 extract_supervised_arrays")


def fit_table_standardizer(
    train_frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.Series, pd.Series]:
    raise NotImplementedError("请完成 fit_table_standardizer")


def transform_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    means: pd.Series,
    scales: pd.Series,
) -> np.ndarray:
    raise NotImplementedError("请完成 transform_features")


def prediction_frame(sample_ids: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    raise NotImplementedError("请完成 prediction_frame")


def save_predictions(path: str | Path, frame: pd.DataFrame) -> None:
    raise NotImplementedError("请完成 save_predictions")
