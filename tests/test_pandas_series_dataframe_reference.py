import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "04_pandas_basics" / "01_series_dataframe" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("pandas_objects_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_series_and_frame_creation() -> None:
    series = solution.make_score_series(["小林", "小周"], [78, 91])
    frame = solution.make_student_frame(
        ["小林", "小周"], [2.5, 4.0], [78, 91]
    )

    assert isinstance(series, pd.Series)
    assert series.name == "score"
    assert series.index.tolist() == ["小林", "小周"]
    assert series.dtype == np.float64
    assert frame.columns.tolist() == ["name", "study_hours", "score"]
    assert frame.shape == (2, 3)
    assert frame["study_hours"].dtype == np.float64
    assert frame["score"].dtype == np.float64


def test_description_selection_and_numpy_copy() -> None:
    frame = solution.make_student_frame(
        ["小林", "小周"], [2.5, 4.0], [78, 91]
    )
    description = solution.describe_frame(frame)
    selected = solution.select_columns(frame, ["score", "study_hours"])
    array = solution.series_to_float_array(frame["score"])
    array[0] = -1

    assert description == {
        "shape": (2, 3),
        "columns": ["name", "study_hours", "score"],
        "dtypes": {"name": "object", "study_hours": "float64", "score": "float64"},
    }
    assert selected.columns.tolist() == ["score", "study_hours"]
    assert selected.shape == (2, 2)
    assert frame.loc[0, "score"] == 78.0


def test_invalid_lengths_and_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError):
        solution.make_score_series(["小林"], [78, 91])
    frame = solution.make_student_frame(["小林"], [2.5], [78])
    with pytest.raises(KeyError):
        solution.select_columns(frame, ["unknown"])
