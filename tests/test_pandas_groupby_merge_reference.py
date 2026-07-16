import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "04_pandas_basics" / "04_groupby_merge"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("pandas_groupby_merge_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def load_tables():
    return solution.read_students(TOPIC / "data" / "students.csv"), solution.read_scores(
        TOPIC / "data" / "scores.csv"
    )


def test_course_summary_has_plain_index_and_named_columns() -> None:
    _, scores = load_tables()
    summary = solution.summarize_courses(scores)
    assert summary.columns.tolist() == [
        "course",
        "record_count",
        "student_count",
        "mean_score",
        "max_score",
    ]
    assert isinstance(summary.index, pd.RangeIndex)
    assert summary.to_dict("records") == [
        {"course": "math", "record_count": 3, "student_count": 3, "mean_score": pytest.approx(190 / 3), "max_score": 80.0},
        {"course": "python", "record_count": 3, "student_count": 3, "mean_score": pytest.approx(200 / 3), "max_score": 90.0},
    ]


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_merge_modes_are_audited_and_relationship_is_many_to_one(how: str) -> None:
    students, scores = load_tables()
    merged, audit = solution.merge_students_and_scores(students, scores, how=how)
    assert len(merged) == 6
    assert audit == {"both": 6, "left_only": 0, "right_only": 0}


def test_merge_audit_exposes_unknown_score_student_and_student_without_score() -> None:
    students, scores = load_tables()
    students = pd.concat(
        [students, pd.DataFrame([["S005", "2025", "Nanjing"]], columns=students.columns)],
        ignore_index=True,
    )
    scores = pd.concat(
        [scores, pd.DataFrame([["S999", "math", 70.0]], columns=scores.columns)],
        ignore_index=True,
    )
    merged, audit = solution.merge_students_and_scores(students, scores, how="outer")
    assert len(merged) == 8
    assert audit == {"both": 6, "left_only": 1, "right_only": 1}


def test_many_to_many_row_count_is_predicted_by_key_products() -> None:
    left = pd.DataFrame({"key": ["A", "A", "B"], "x": [1, 2, 3]})
    right = pd.DataFrame({"key": ["A", "A", "A", "C"], "y": [1, 2, 3, 4]})
    expected = solution.predict_inner_merge_rows(left, right, "key")
    actual = len(left.merge(right, on="key", how="inner"))
    assert expected == actual == 6


def test_duplicate_student_key_is_rejected_before_merge() -> None:
    students, scores = load_tables()
    duplicated = pd.concat([students, students.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="必须唯一"):
        solution.merge_students_and_scores(duplicated, scores)


def test_end_to_end_report_and_strict_bytes(tmp_path: Path) -> None:
    students, scores = load_tables()
    report = solution.build_cohort_course_report(students, scores)
    output = tmp_path / "nested" / "report.csv"
    solution.save_report(output, report)
    assert report.shape == (4, 6)
    assert output.read_bytes() == (TOPIC / "expected" / "cohort_course_report.csv").read_bytes()


def test_strict_report_rejects_unknown_student() -> None:
    students, scores = load_tables()
    extra = pd.DataFrame([["UNKNOWN", "math", 80.0]], columns=scores.columns)
    with pytest.raises(ValueError, match="未知student_id"):
        solution.build_cohort_course_report(students, pd.concat([scores, extra], ignore_index=True))


def test_functions_do_not_modify_input_tables() -> None:
    students, scores = load_tables()
    original_students = students.copy(deep=True)
    original_scores = scores.copy(deep=True)
    solution.summarize_courses(scores)
    solution.build_cohort_course_report(students, scores)
    pd.testing.assert_frame_equal(students, original_students)
    pd.testing.assert_frame_equal(scores, original_scores)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.predict_inner_merge_rows(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}), "key"),
        lambda: solution.merge_students_and_scores(*load_tables(), how="cross"),
        lambda: solution.read_students(TOPIC / "data" / "missing.csv"),
        lambda: solution.save_report(Path("unused.csv"), pd.DataFrame()),
    ],
)
def test_invalid_groupby_merge_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError, FileNotFoundError)):
        call()
