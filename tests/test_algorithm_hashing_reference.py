import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "algorithms" / "03_hashing_sequences"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("hashing_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_frequency_and_unique_preserve_first_occurrence_order() -> None:
    values = [3, 7, 3, 2, 7, 3]
    assert solution.frequency_in_first_order(values) == [(3, 3), (7, 2), (2, 1)]
    assert solution.stable_unique(values) == [3, 7, 2]


def test_hash_functions_do_not_modify_input() -> None:
    values = [3, 7, 3, 2]
    original = values.copy()
    solution.frequency_in_first_order(values)
    solution.stable_unique(values)
    solution.first_two_sum_pair(values, 10)
    assert values == original


def test_two_equal_values_use_two_different_indices() -> None:
    assert solution.first_two_sum_pair([3, 3], 6) == (0, 1)
    assert solution.first_two_sum_pair([3], 6) is None


def test_first_pair_rule_uses_earliest_right_then_earliest_left() -> None:
    assert solution.first_two_sum_pair([4, 1, 6, 9, 4], 10) == (0, 2)
    assert solution.first_two_sum_pair([1, 1, 9], 10) == (0, 2)


@pytest.mark.parametrize(
    "values,target,expected",
    [
        ([], 0, None),
        ([0, -2, 5, 2], 0, (1, 3)),
        ([5, 5, 5], 10, (0, 1)),
        ([1, 2, 3], 100, None),
    ],
)
def test_two_sum_boundaries(values, target, expected) -> None:
    assert solution.first_two_sum_pair(values, target) == expected


def test_sample_solve_matches_expected() -> None:
    source = (TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (TOPIC / "expected" / "sample_output.txt").read_text(encoding="utf-8").rstrip("\n")
    assert solution.solve(source) == expected


def test_empty_sequence_has_deterministic_output() -> None:
    assert solution.solve("0 5\n\n") == "unique:\nfrequencies:\npair: NONE"


def test_stdin_and_file_cli(tmp_path: Path) -> None:
    source = (TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (TOPIC / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    stdin_run = subprocess.run(
        [sys.executable, str(SOLUTION)],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    assert stdin_run.stdout == expected
    output = tmp_path / "nested" / "hash.txt"
    subprocess.run(
        [sys.executable, str(SOLUTION), str(TOPIC / "data" / "sample_input.txt"), str(output)],
        check=True,
    )
    assert output.read_bytes() == (TOPIC / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    ["", "3 5\n1 2\n", "2 x\n1 2\n", "-1 5\n\n", "2 5\n1 bad\n", "2 5 extra\n1 2\n"],
)
def test_invalid_problem_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        solution.parse_problem(text)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.frequency_in_first_order([1, "2"]),
        lambda: solution.first_two_sum_pair([1, 2], True),
    ],
)
def test_invalid_hash_function_inputs_are_rejected(call) -> None:
    with pytest.raises(TypeError):
        call()
