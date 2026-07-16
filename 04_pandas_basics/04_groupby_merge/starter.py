"""学生练习：pandas分组聚合、连接和严格报告。"""

from pathlib import Path

import pandas as pd


def read_students(path: str | Path) -> pd.DataFrame:
    raise NotImplementedError("请完成 read_students")


def read_scores(path: str | Path) -> pd.DataFrame:
    raise NotImplementedError("请完成 read_scores")


def summarize_courses(scores: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("请完成 summarize_courses")


def predict_inner_merge_rows(
    left: pd.DataFrame, right: pd.DataFrame, key: str
) -> int:
    raise NotImplementedError("请完成 predict_inner_merge_rows")


def merge_students_and_scores(
    students: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    how: str = "left",
) -> tuple[pd.DataFrame, dict[str, int]]:
    raise NotImplementedError("请完成 merge_students_and_scores")


def build_cohort_course_report(
    students: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    raise NotImplementedError("请完成 build_cohort_course_report")


def save_report(path: str | Path, report: pd.DataFrame) -> None:
    raise NotImplementedError("请完成 save_report")
