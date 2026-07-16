import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "04_pandas_basics" / "03_cleaning"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("pandas_cleaning_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_cleaning_pipeline_and_strict_output(tmp_path: Path) -> None:
    raw = solution.read_raw_table(TOPIC / "data" / "dirty_students.csv")
    normalized = solution.normalize_and_convert(raw)
    deduplicated = solution.drop_duplicate_students(normalized)
    medians = solution.fit_numeric_medians(deduplicated)
    filled = solution.fill_numeric_missing(deduplicated, medians)
    output = tmp_path / "nested" / "cleaned.csv"
    solution.save_cleaned_table(output, filled)

    assert raw.shape == (5, 4)
    assert raw["study_hours"].dtype == object
    assert normalized.loc[0, "name"] == "小林"
    assert np.isnan(normalized.loc[3, "study_hours"])
    assert deduplicated["student_id"].tolist() == ["S001", "S002", "S003", "S004"]
    assert medians.to_dict() == {"study_hours": 4.0, "attendance": 0.9}
    assert not filled.isna().any().any()
    assert output.read_bytes() == (
        TOPIC / "expected" / "cleaned_students.csv"
    ).read_bytes()


def test_transform_uses_supplied_training_medians() -> None:
    test_frame = pd.DataFrame(
        {
            "student_id": ["T001", "T002"],
            "name": ["甲", "乙"],
            "study_hours": [np.nan, 1000.0],
            "attendance": [100.0, np.nan],
        }
    )
    training_medians = pd.Series(
        {"study_hours": 4.0, "attendance": 0.9}, dtype=float
    )
    result = solution.fill_numeric_missing(test_frame, training_medians)

    assert result.loc[0, "study_hours"] == 4.0
    assert result.loc[1, "attendance"] == 0.9
    assert result.loc[1, "study_hours"] == 1000.0


def test_functions_do_not_modify_input() -> None:
    raw = solution.read_raw_table(TOPIC / "data" / "dirty_students.csv")
    original = raw.copy(deep=True)
    solution.normalize_and_convert(raw)
    pd.testing.assert_frame_equal(raw, original)


def test_all_missing_numeric_column_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "student_id": ["S1", "S2"],
            "name": ["甲", "乙"],
            "study_hours": [np.nan, np.nan],
            "attendance": [0.8, 0.9],
        }
    )
    with pytest.raises(ValueError, match="无法拟合"):
        solution.fit_numeric_medians(frame)


def test_missing_file_and_wrong_columns_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        solution.read_raw_table(tmp_path / "missing.csv")
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("student_id,name\nS1,甲\n", encoding="utf-8")
    with pytest.raises(ValueError, match="列必须"):
        solution.read_raw_table(wrong)
