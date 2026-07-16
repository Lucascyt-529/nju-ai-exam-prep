import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SORT_TOPIC = ROOT / "algorithms" / "01_sorting_records"
BINARY_TOPIC = ROOT / "algorithms" / "02_binary_search_boundaries"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sorting = load_module("sorting_solution", SORT_TOPIC / "reference" / "solution.py")
binary = load_module("binary_solution", BINARY_TOPIC / "reference" / "solution.py")


def test_merge_sort_is_descending_stable_and_non_mutating() -> None:
    records = [("first", 80), ("high", 95), ("second", 80), ("low", 70)]
    original = records.copy()
    result = sorting.stable_merge_sort_records(records)
    assert result == [("high", 95), ("first", 80), ("second", 80), ("low", 70)]
    assert records == original


def test_sorting_sample_solve_matches_expected_bytes() -> None:
    source = (SORT_TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (SORT_TOPIC / "expected" / "sample_output.txt").read_text(encoding="utf-8").rstrip("\n")
    assert sorting.solve(source) == expected


def test_sorting_stdin_cli() -> None:
    source = (SORT_TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (SORT_TOPIC / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SORT_TOPIC / "reference" / "solution.py")],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout == expected


def test_sorting_file_cli(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "sorted.txt"
    subprocess.run(
        [
            sys.executable,
            str(SORT_TOPIC / "reference" / "solution.py"),
            str(SORT_TOPIC / "data" / "sample_input.txt"),
            str(output),
        ],
        check=True,
    )
    assert output.read_bytes() == (SORT_TOPIC / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    ["", "2\nA 1\n", "1\nA bad\n", "2\nA 1\nA 2\n", "1 extra\nA 1\n"],
)
def test_invalid_sorting_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        sorting.parse_records(text)


@pytest.mark.parametrize(
    "values,target,expected_lower,expected_upper",
    [
        ([1, 2, 2, 2, 5], 2, 1, 4),
        ([1, 2, 2, 2, 5], 3, 4, 4),
        ([1, 2, 2, 2, 5], -10, 0, 0),
        ([1, 2, 2, 2, 5], 10, 5, 5),
        ([], 3, 0, 0),
    ],
)
def test_binary_boundaries(values, target, expected_lower, expected_upper) -> None:
    assert binary.lower_bound(values, target) == expected_lower
    assert binary.upper_bound(values, target) == expected_upper


def test_binary_sample_solve_matches_expected() -> None:
    source = (BINARY_TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (BINARY_TOPIC / "expected" / "sample_output.txt").read_text(encoding="utf-8").rstrip("\n")
    assert binary.solve(source) == expected


def test_binary_empty_array_problem_is_supported() -> None:
    values, queries = binary.parse_problem("0 2\n\n-1 5\n")
    assert values == [] and queries == [-1, 5]
    assert binary.solve("0 2\n\n-1 5\n") == "-1 0 0 0\n5 0 0 0"


def test_binary_stdin_and_file_cli(tmp_path: Path) -> None:
    source = (BINARY_TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (BINARY_TOPIC / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    stdin_run = subprocess.run(
        [sys.executable, str(BINARY_TOPIC / "reference" / "solution.py")],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    assert stdin_run.stdout == expected
    output = tmp_path / "binary" / "answer.txt"
    subprocess.run(
        [
            sys.executable,
            str(BINARY_TOPIC / "reference" / "solution.py"),
            str(BINARY_TOPIC / "data" / "sample_input.txt"),
            str(output),
        ],
        check=True,
    )
    assert output.read_bytes() == (BINARY_TOPIC / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "3 1\n1 3 2\n2\n",
        "2 1\n1\n2\n",
        "2 0\n1 2\n\n",
        "2 1\n1 x\n2\n",
    ],
)
def test_invalid_binary_problem_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        binary.parse_problem(text)
