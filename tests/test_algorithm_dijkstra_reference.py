import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "algorithms" / "06_dijkstra_shortest_path"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("dijkstra_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_sample_distances_path_and_stale_entries() -> None:
    text = (TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    adjacency, start, goal = solution.parse_weighted_graph(text)
    distances, parents, stale_entries = solution.dijkstra(adjacency, start)
    assert distances[goal] == 7
    assert solution.reconstruct_path(parents, start, goal) == [0, 2, 1, 3, 4]
    assert stale_entries >= 1


def test_zero_weight_edges_and_large_integer_distance() -> None:
    text = "4 3 0 3\n0 1 0\n1 2 1000000000000\n2 3 0\n"
    adjacency, start, goal = solution.parse_weighted_graph(text)
    distances, parents, _ = solution.dijkstra(adjacency, start)
    assert distances[goal] == 1000000000000
    assert solution.reconstruct_path(parents, start, goal) == [0, 1, 2, 3]


def test_equal_paths_keep_first_deterministic_parent() -> None:
    first = "4 4 0 3\n0 2 1\n2 3 1\n0 1 1\n1 3 1\n"
    second = "4 4 0 3\n1 3 1\n0 1 1\n2 3 1\n0 2 1\n"
    assert solution.solve(first) == solution.solve(second) == "distance: 2\npath: 0 1 3"


def test_unreachable_and_start_equals_goal() -> None:
    assert solution.solve("3 1 0 2\n0 1 5\n") == "distance: -1\npath: NONE"
    assert solution.solve("2 0 1 1\n") == "distance: 0\npath: 1"


def test_sample_output_and_both_cli_modes(tmp_path: Path) -> None:
    source = (TOPIC / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (TOPIC / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    assert solution.solve(source) == expected.rstrip("\n")
    stdin_run = subprocess.run(
        [sys.executable, str(SOLUTION)], input=source, text=True, capture_output=True, check=True
    )
    assert stdin_run.stdout == expected
    output = tmp_path / "nested" / "dijkstra.txt"
    subprocess.run(
        [sys.executable, str(SOLUTION), str(TOPIC / "data" / "sample_input.txt"), str(output)], check=True
    )
    assert output.read_bytes() == (TOPIC / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "3 1 0 3\n0 1 2\n",
        "3 2 0 2\n0 1 1\n",
        "3 1 0 2\n0 1 -1\n",
        "3 1 0 2\n1 1 0\n",
        "3 2 0 2\n0 1 1\n0 1 2\n",
    ],
)
def test_invalid_weighted_graph_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        solution.parse_weighted_graph(text)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.dijkstra([[(1, -1)], []], 0),
        lambda: solution.dijkstra([], 0),
        lambda: solution.reconstruct_path([], 0, 0),
    ],
)
def test_invalid_dijkstra_function_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()
