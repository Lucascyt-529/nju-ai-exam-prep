"""参考实现：pandas分组聚合、连接和严格报告。"""

from pathlib import Path

import numpy as np
import pandas as pd


STUDENT_COLUMNS = ["student_id", "cohort", "city"]
SCORE_COLUMNS = ["student_id", "course", "score"]
REPORT_COLUMNS = [
    "cohort",
    "course",
    "record_count",
    "student_count",
    "mean_score",
    "pass_rate",
]
VALID_HOW = {"inner", "left", "right", "outer"}


def _validate_students(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.columns.tolist() != STUDENT_COLUMNS:
        raise ValueError(f"学生表列必须按顺序为{STUDENT_COLUMNS}")
    if frame.empty:
        raise ValueError("学生表不能为空")
    if frame["student_id"].isna().any() or frame["student_id"].astype(str).str.strip().eq("").any():
        raise ValueError("student_id不能为空")
    if frame["student_id"].duplicated().any():
        raise ValueError("学生表student_id必须唯一")
    if frame[["cohort", "city"]].isna().any().any():
        raise ValueError("cohort和city不能为空")


def _validate_scores(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.columns.tolist() != SCORE_COLUMNS:
        raise ValueError(f"成绩表列必须按顺序为{SCORE_COLUMNS}")
    if frame.empty:
        raise ValueError("成绩表不能为空")
    if frame[["student_id", "course"]].isna().any().any():
        raise ValueError("student_id和course不能为空")
    if frame["student_id"].astype(str).str.strip().eq("").any() or frame["course"].astype(str).str.strip().eq("").any():
        raise ValueError("student_id和course不能为空字符串")
    numeric = pd.to_numeric(frame["score"], errors="coerce")
    if numeric.isna().any() or not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise ValueError("score必须是有限数值")
    if ((numeric < 0) | (numeric > 100)).any():
        raise ValueError("score必须位于[0,100]")


def read_students(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到学生表: {source}")
    frame = pd.read_csv(source, dtype={"student_id": str, "cohort": str, "city": str})
    _validate_students(frame)
    return frame


def read_scores(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到成绩表: {source}")
    frame = pd.read_csv(source, dtype={"student_id": str, "course": str})
    _validate_scores(frame)
    result = frame.copy()
    result["score"] = pd.to_numeric(result["score"]).astype(float)
    return result


def summarize_courses(scores: pd.DataFrame) -> pd.DataFrame:
    _validate_scores(scores)
    return (
        scores.groupby("course", dropna=False, sort=True)
        .agg(
            record_count=("score", "size"),
            student_count=("student_id", "nunique"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
        )
        .reset_index()
    )


def predict_inner_merge_rows(
    left: pd.DataFrame, right: pd.DataFrame, key: str
) -> int:
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise TypeError("left和right必须是DataFrame")
    if key not in left.columns or key not in right.columns:
        raise ValueError("连接键必须同时存在于左右表")
    if left[key].isna().any() or right[key].isna().any():
        raise ValueError("行数审计不接受缺失连接键")
    left_counts = left.groupby(key, sort=False).size()
    right_counts = right.groupby(key, sort=False).size()
    common = left_counts.index.intersection(right_counts.index)
    return int(sum(int(left_counts[value]) * int(right_counts[value]) for value in common))


def merge_students_and_scores(
    students: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    how: str = "left",
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_students(students)
    _validate_scores(scores)
    if how not in VALID_HOW:
        raise ValueError(f"how必须属于{sorted(VALID_HOW)}")
    merged = scores.merge(
        students,
        on="student_id",
        how=how,
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    counts = merged["_merge"].value_counts().reindex(
        ["both", "left_only", "right_only"], fill_value=0
    )
    audit = {name: int(value) for name, value in counts.items()}
    return merged, audit


def build_cohort_course_report(
    students: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    merged, audit = merge_students_and_scores(students, scores, how="left")
    if audit["left_only"] != 0:
        unknown = sorted(merged.loc[merged["_merge"] == "left_only", "student_id"].unique())
        raise ValueError(f"成绩表存在未知student_id: {unknown}")
    matched = merged.drop(columns="_merge")
    report = (
        matched.groupby(["cohort", "course"], sort=True, dropna=False)
        .agg(
            record_count=("score", "size"),
            student_count=("student_id", "nunique"),
            mean_score=("score", "mean"),
            pass_rate=("score", lambda values: float((values >= 60).mean())),
        )
        .reset_index()
        .sort_values(["cohort", "course"], kind="stable")
        .reset_index(drop=True)
    )
    return report[REPORT_COLUMNS]


def save_report(path: str | Path, report: pd.DataFrame) -> None:
    if not isinstance(report, pd.DataFrame) or report.columns.tolist() != REPORT_COLUMNS:
        raise ValueError(f"报告列必须按顺序为{REPORT_COLUMNS}")
    if report.empty or report.isna().any().any():
        raise ValueError("报告不能为空或包含缺失值")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        destination,
        index=False,
        columns=REPORT_COLUMNS,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
