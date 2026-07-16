import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "04_pandas_basics" / "06_sort_pivot_melt"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("pandas_reshape_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def long_frame():
    return pd.read_csv(TOPIC / "data" / "long_scores.csv")


def test_stable_sort_preserves_tie_order_and_can_reset_index() -> None:
    frame = pd.DataFrame(
        {"group": ["A", "A", "A"], "score": [80, 80, 90], "order": ["first", "second", "third"]},
        index=[10, 20, 30],
    )
    kept = solution.stable_sort_records(frame, ["score"], [False], reset_index=False)
    reset = solution.stable_sort_records(frame, ["score"], [False], reset_index=True)
    assert kept["order"].tolist() == ["third", "first", "second"]
    assert kept.index.tolist() == [30, 10, 20]
    assert reset.index.tolist() == [0, 1, 2]


def test_multi_column_sort_uses_matching_ascending_flags() -> None:
    frame = pd.DataFrame({"cohort": [2025, 2024, 2024], "score": [90, 60, 80]})
    result = solution.stable_sort_records(frame, ["cohort", "score"], [True, False])
    assert result.to_dict("records") == [
        {"cohort": 2024, "score": 80},
        {"cohort": 2024, "score": 60},
        {"cohort": 2025, "score": 90},
    ]


def test_strict_pivot_has_deterministic_rows_columns_and_metadata() -> None:
    wide = solution.strict_long_to_wide(
        long_frame(), index="student_id", columns="course", values="score"
    )
    assert wide.columns.tolist() == ["student_id", "math", "python"]
    assert wide.columns.name is None
    assert isinstance(wide.index, pd.RangeIndex)
    assert wide.to_dict("records") == [
        {"student_id": "S001", "math": 80, "python": 90},
        {"student_id": "S002", "math": 60, "python": 70},
    ]


def test_strict_pivot_rejects_duplicate_cell_keys() -> None:
    frame = pd.concat([long_frame(), long_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="pivot键不唯一"):
        solution.strict_long_to_wide(frame, index="student_id", columns="course", values="score")


@pytest.mark.parametrize(
    "aggfunc, expected",
    [("mean", 75.0), ("sum", 150), ("max", 80), ("min", 70)],
)
def test_pivot_table_requires_explicit_duplicate_aggregation(aggfunc: str, expected: float) -> None:
    frame = pd.DataFrame(
        {"student_id": ["S1", "S1"], "course": ["math", "math"], "score": [70, 80]}
    )
    wide = solution.aggregate_long_to_wide(
        frame, index="student_id", columns="course", values="score", aggfunc=aggfunc
    )
    assert wide.loc[0, "math"] == expected


def test_missing_combination_becomes_nan_and_melt_can_drop_it() -> None:
    incomplete = long_frame().query("not (student_id == 'S002' and course == 'python')")
    wide = solution.strict_long_to_wide(
        incomplete, index="student_id", columns="course", values="score"
    )
    assert pd.isna(wide.loc[wide["student_id"] == "S002", "python"]).item()
    dropped = solution.wide_to_long(
        wide, index="student_id", variable_name="course", value_name="score", drop_missing=True
    )
    kept = solution.wide_to_long(
        wide, index="student_id", variable_name="course", value_name="score", drop_missing=False
    )
    assert len(dropped) == 3 and len(kept) == 4


def test_complete_long_wide_long_round_trip() -> None:
    original = long_frame().sort_values(["student_id", "course"], kind="stable").reset_index(drop=True)
    wide = solution.strict_long_to_wide(
        original, index="student_id", columns="course", values="score"
    )
    restored = solution.wide_to_long(
        wide, index="student_id", variable_name="course", value_name="score"
    )
    pd.testing.assert_frame_equal(restored, original)


def test_strict_wide_output_bytes(tmp_path: Path) -> None:
    wide = solution.strict_long_to_wide(
        long_frame(), index="student_id", columns="course", values="score"
    )
    output = tmp_path / "nested" / "wide.csv"
    solution.save_wide_table(output, wide)
    assert output.read_bytes() == (TOPIC / "expected" / "wide_scores.csv").read_bytes()


def test_functions_do_not_modify_input() -> None:
    frame = long_frame()
    original = frame.copy(deep=True)
    solution.stable_sort_records(frame, ["score"], [True])
    solution.strict_long_to_wide(frame, index="student_id", columns="course", values="score")
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.stable_sort_records(long_frame(), ["score"], [True, False]),
        lambda: solution.aggregate_long_to_wide(
            long_frame(), index="student_id", columns="course", values="score", aggfunc="first"
        ),
        lambda: solution.wide_to_long(
            pd.DataFrame({"student_id": ["S1"], "math": [80]}),
            index="student_id",
            variable_name="math",
            value_name="score",
        ),
        lambda: solution.save_wide_table(Path("unused.csv"), pd.DataFrame()),
    ],
)
def test_invalid_sort_reshape_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()
