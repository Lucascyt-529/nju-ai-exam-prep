import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
KNAPSACK = ROOT / "algorithms" / "09_zero_one_knapsack"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


knapsack = load_module("knapsack_solution", KNAPSACK / "reference" / "solution.py")


def test_sample_value_table_and_reconstruction() -> None:
    maximum, selected, table = knapsack.zero_one_knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7)
    assert maximum == 9
    assert selected == [1, 2]
    assert len(table) == 5
    assert all(len(row) == 8 for row in table)
    assert table[-1][-1] == maximum


def test_zero_capacity_and_no_items() -> None:
    assert knapsack.zero_one_knapsack([2, 3], [5, 8], 0)[:2] == (0, [])
    assert knapsack.zero_one_knapsack([], [], 5)[:2] == (0, [])
    assert knapsack.solve("0 5\n") == "maximum: 0\nweight: 0\nitems: NONE"


def test_no_item_fits() -> None:
    maximum, selected, table = knapsack.zero_one_knapsack([4, 5], [6, 10], 3)
    assert (maximum, selected) == (0, [])
    assert table[-1] == [0, 0, 0, 0]


def test_each_item_can_be_used_at_most_once() -> None:
    maximum, selected, _ = knapsack.zero_one_knapsack([2], [3], 4)
    assert (maximum, selected) == (3, [0])


def test_value_tie_excludes_later_item() -> None:
    assert knapsack.zero_one_knapsack([2, 2], [5, 5], 2)[:2] == (5, [0])
    assert knapsack.zero_one_knapsack([2, 1, 1], [2, 1, 1], 2)[:2] == (2, [0])


def test_inputs_are_not_modified() -> None:
    weights = [5, 1, 3]
    values = [7, 1, 4]
    original_weights = weights.copy()
    original_values = values.copy()
    knapsack.zero_one_knapsack(weights, values, 4)
    assert weights == original_weights
    assert values == original_values


@pytest.mark.parametrize(
    "weights, values, capacity",
    [
        ((1, 2), [3, 4], 2),
        ([1, 2], (3, 4), 2),
        ([0, 2], [3, 4], 2),
        ([True, 2], [3, 4], 2),
        ([1, 2], [3, -1], 2),
        ([1, 2], [3], 2),
        ([1, 2], [3, 4], -1),
        ([1, 2], [3, 4], True),
    ],
)
def test_invalid_function_inputs_are_rejected(weights, values, capacity) -> None:
    with pytest.raises(ValueError):
        knapsack.zero_one_knapsack(weights, values, capacity)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2 5\n1 2\n",
        "-1 5\n",
        "1 -1\n1 2\n",
        "1 5\n0 2\n",
        "1 5\n1 -2\n",
        "1 5\n1 2 3\n",
        "x 5\n1 2\n",
        "1 5\n1 x\n",
        "1 5\n1 2\n2 3\n",
    ],
)
def test_invalid_text_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        knapsack.parse_problem(text)


def test_reconstruction_rejects_wrong_table_shape_or_transition() -> None:
    with pytest.raises(ValueError):
        knapsack.reconstruct_selected_items([1], [2], 1, [[0]])
    with pytest.raises(ValueError):
        knapsack.reconstruct_selected_items([1], [2], 1, [[0, 0], [0, 1]])


def test_sample_solve_and_cli_modes(tmp_path: Path) -> None:
    source = (KNAPSACK / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (KNAPSACK / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    assert knapsack.solve(source) == expected.rstrip("\n")

    stdin_run = subprocess.run(
        [sys.executable, str(KNAPSACK / "reference" / "solution.py")],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )
    assert stdin_run.stdout == expected

    output = tmp_path / "knapsack" / "answer.txt"
    subprocess.run(
        [
            sys.executable,
            str(KNAPSACK / "reference" / "solution.py"),
            str(KNAPSACK / "data" / "sample_input.txt"),
            str(output),
        ],
        check=True,
    )
    assert output.read_bytes() == (KNAPSACK / "expected" / "sample_output.txt").read_bytes()
