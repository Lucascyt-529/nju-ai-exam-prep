"""参考实现：网格BFS最短路与路径恢复。"""

import argparse
from collections import deque
from pathlib import Path
import sys


Cell = tuple[int, int]
DIRECTIONS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def parse_grid(text: str) -> tuple[list[str], Cell, Cell]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    lines = text.strip().splitlines()
    first = lines[0].split()
    if len(first) != 2:
        raise ValueError("第一行必须是rows cols")
    try:
        rows, columns = map(int, first)
    except ValueError as error:
        raise ValueError("rows和cols必须是整数") from error
    if rows <= 0 or columns <= 0 or len(lines) != rows + 3:
        raise ValueError("网格规模或输入行数无效")
    grid = lines[1 : rows + 1]
    if any(len(row) != columns or any(value not in ".#" for value in row) for row in grid):
        raise ValueError("网格每行长度必须正确且只能包含.或#")
    try:
        start_parts = list(map(int, lines[rows + 1].split()))
        goal_parts = list(map(int, lines[rows + 2].split()))
    except ValueError as error:
        raise ValueError("起点和终点坐标必须是整数") from error
    if len(start_parts) != 2 or len(goal_parts) != 2:
        raise ValueError("起点和终点必须各有两个坐标")
    start = (start_parts[0], start_parts[1])
    goal = (goal_parts[0], goal_parts[1])
    for name, (row, column) in (("起点", start), ("终点", goal)):
        if not 0 <= row < rows or not 0 <= column < columns or grid[row][column] == "#":
            raise ValueError(f"{name}越界或位于障碍")
    return grid, start, goal


def shortest_grid_path(grid: list[str], start: Cell, goal: Cell) -> tuple[int, list[Cell]]:
    if not isinstance(grid, list):
        raise ValueError("grid必须是列表")
    rows = len(grid)
    columns = len(grid[0]) if rows else 0
    if (
        rows == 0
        or columns == 0
        or any(not isinstance(row, str) or len(row) != columns or any(value not in ".#" for value in row) for row in grid)
    ):
        raise ValueError("grid必须是非空矩形且只能包含.或#")
    for name, cell in (("start", start), ("goal", goal)):
        if (
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in cell)
            or not 0 <= cell[0] < rows
            or not 0 <= cell[1] < columns
            or grid[cell[0]][cell[1]] == "#"
        ):
            raise ValueError(f"{name}坐标无效或位于障碍")
    if start == goal:
        return 0, [start]
    queue = deque([start])
    parents: dict[Cell, Cell | None] = {start: None}
    while queue:
        row, column = queue.popleft()
        for row_delta, column_delta in DIRECTIONS:
            neighbor = (row + row_delta, column + column_delta)
            next_row, next_column = neighbor
            if (
                0 <= next_row < rows
                and 0 <= next_column < columns
                and grid[next_row][next_column] == "."
                and neighbor not in parents
            ):
                parents[neighbor] = (row, column)
                if neighbor == goal:
                    path = [goal]
                    current = goal
                    while parents[current] is not None:
                        current = parents[current]
                        path.append(current)
                    path.reverse()
                    return len(path) - 1, path
                queue.append(neighbor)
    return -1, []


def solve(text: str) -> str:
    grid, start, goal = parse_grid(text)
    distance, path = shortest_grid_path(grid, start, goal)
    path_text = "NONE" if not path else " ".join(f"{row},{column}" for row, column in path)
    return f"distance: {distance}\npath: {path_text}"


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="网格BFS最短路")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("output_path", nargs="?")
    args = parser.parse_args(argv)
    text = Path(args.input_path).read_text(encoding="utf-8") if args.input_path else sys.stdin.read()
    output = solve(text) + "\n"
    if args.output_path:
        destination = Path(args.output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8", newline="")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    run_cli()
