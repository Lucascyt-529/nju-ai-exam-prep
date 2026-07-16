import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "algorithms" / "04_graph_traversal"
GRID = ROOT / "algorithms" / "05_grid_shortest_path"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


graph = load_module("graph_solution", GRAPH / "reference" / "solution.py")
grid = load_module("grid_solution", GRID / "reference" / "solution.py")


def test_graph_sample_and_sorted_adjacency() -> None:
    text = (GRAPH / "data" / "sample_input.txt").read_text(encoding="utf-8")
    adjacency, start = graph.parse_graph(text)
    assert adjacency[0] == [1, 2] and start == 0
    assert graph.bfs(adjacency, start) == [0, 1, 2, 3]
    assert graph.dfs(adjacency, start) == [0, 1, 3, 2]
    assert graph.connected_components(adjacency) == [[0, 1, 2, 3], [4, 5]]


def test_graph_edge_input_order_does_not_change_traversal() -> None:
    first = "4 3 0\n0 2\n1 3\n0 1\n"
    second = "4 3 0\n0 1\n0 2\n3 1\n"
    assert graph.solve(first) == graph.solve(second)


def test_graph_isolated_vertices_are_components() -> None:
    adjacency, _ = graph.parse_graph("4 1 0\n0 1\n")
    assert graph.connected_components(adjacency) == [[0, 1], [2], [3]]


def test_graph_sample_output_and_cli(tmp_path: Path) -> None:
    source = (GRAPH / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (GRAPH / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    assert graph.solve(source) == expected.rstrip("\n")
    stdin_run = subprocess.run(
        [sys.executable, str(GRAPH / "reference" / "solution.py")], input=source, text=True, capture_output=True, check=True
    )
    assert stdin_run.stdout == expected
    output = tmp_path / "graph" / "answer.txt"
    subprocess.run(
        [sys.executable, str(GRAPH / "reference" / "solution.py"), str(GRAPH / "data" / "sample_input.txt"), str(output)], check=True
    )
    assert output.read_bytes() == (GRAPH / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    ["", "3 1 3\n0 1\n", "3 2 0\n0 1\n", "3 1 0\n1 1\n", "3 2 0\n0 1\n1 0\n"],
)
def test_invalid_graph_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        graph.parse_graph(text)


def test_grid_sample_distance_path_and_output() -> None:
    source = (GRID / "data" / "sample_input.txt").read_text(encoding="utf-8")
    parsed_grid, start, goal = grid.parse_grid(source)
    distance, path = grid.shortest_grid_path(parsed_grid, start, goal)
    assert distance == 7 and len(path) == 8
    assert path[0] == start and path[-1] == goal
    expected = (GRID / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    assert grid.solve(source) == expected.rstrip("\n")


def test_grid_start_equals_goal() -> None:
    parsed_grid, start, goal = grid.parse_grid("1 1\n.\n0 0\n0 0\n")
    assert grid.shortest_grid_path(parsed_grid, start, goal) == (0, [(0, 0)])


def test_grid_unreachable_and_deterministic_tie_path() -> None:
    assert grid.solve("2 2\n.#\n#.\n0 0\n1 1\n") == "distance: -1\npath: NONE"
    open_grid, start, goal = grid.parse_grid("2 2\n..\n..\n0 0\n1 1\n")
    assert grid.shortest_grid_path(open_grid, start, goal)[1] == [(0, 0), (0, 1), (1, 1)]


@pytest.mark.parametrize(
    "grid_value,start,goal",
    [
        ([], (0, 0), (0, 0)),
        (["..", "."], (0, 0), (1, 0)),
        ([".X"], (0, 0), (0, 1)),
        ([".#"], (0, 0), (0, 1)),
    ],
)
def test_shortest_path_rejects_invalid_direct_inputs(grid_value, start, goal) -> None:
    with pytest.raises(ValueError):
        grid.shortest_grid_path(grid_value, start, goal)


def test_grid_stdin_and_file_cli(tmp_path: Path) -> None:
    source = (GRID / "data" / "sample_input.txt").read_text(encoding="utf-8")
    expected = (GRID / "expected" / "sample_output.txt").read_text(encoding="utf-8")
    stdin_run = subprocess.run(
        [sys.executable, str(GRID / "reference" / "solution.py")], input=source, text=True, capture_output=True, check=True
    )
    assert stdin_run.stdout == expected
    output = tmp_path / "grid" / "answer.txt"
    subprocess.run(
        [sys.executable, str(GRID / "reference" / "solution.py"), str(GRID / "data" / "sample_input.txt"), str(output)], check=True
    )
    assert output.read_bytes() == (GRID / "expected" / "sample_output.txt").read_bytes()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2 2\n..\n.\n0 0\n1 1\n",
        "1 1\nX\n0 0\n0 0\n",
        "1 1\n#\n0 0\n0 0\n",
        "1 1\n.\n0 0 extra\n0 0\n",
    ],
)
def test_invalid_grid_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        grid.parse_grid(text)
