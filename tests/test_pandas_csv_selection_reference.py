import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "04_pandas_basics" / "02_csv_selection" / "reference" / "solution.py"
DATA = ROOT / "04_pandas_basics" / "02_csv_selection" / "data" / "students.csv"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("pandas_csv_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_load_students_preserves_schema_and_id_strings() -> None:
    frame = solution.load_students(DATA)

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (4, 5)
    assert frame.columns.tolist() == solution.REQUIRED_COLUMNS
    assert frame["student_id"].tolist() == ["S001", "S002", "S003", "S004"]


def test_column_position_and_boolean_selection() -> None:
    frame = solution.load_students(DATA)
    features = solution.select_feature_columns(frame)
    middle = solution.select_rows_by_position(frame, 1, 3)
    high = solution.filter_by_score(frame, 85)

    assert features.shape == (4, 2)
    assert features.columns.tolist() == ["study_hours", "attendance"]
    assert middle["name"].tolist() == ["小周", "小陈"]
    assert high["name"].tolist() == ["小周", "小陈", "小吴"]
    assert high.index.tolist() == [1, 2, 3]


def test_missing_file_and_wrong_schema_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        solution.load_students(tmp_path / "missing.csv")

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("name,score\n小林,78\n", encoding="utf-8")
    with pytest.raises(ValueError):
        solution.load_students(wrong)
