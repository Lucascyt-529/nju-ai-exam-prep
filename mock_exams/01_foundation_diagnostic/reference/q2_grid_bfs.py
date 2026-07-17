"""模拟卷1第2题参考实现。"""

import argparse
from collections import deque
from pathlib import Path
import sys


DIRECTIONS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def load_problem(path: Path) -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("输入为空")
    try:
        rows, columns = map(int, lines[0].split())
    except ValueError as exc:
        raise ValueError("首行必须是行列数") from exc
    if rows <= 0 or columns <= 0 or len(lines) != rows + 3:
        raise ValueError("行列数或输入行数错误")
    grid = lines[1:1 + rows]
    if any(len(row) != columns or set(row) - {".", "#"} for row in grid):
        raise ValueError("网格尺寸或字符错误")
    try:
        start = tuple(map(int, lines[-2].split()))
        goal = tuple(map(int, lines[-1].split()))
    except ValueError as exc:
        raise ValueError("起终点必须是整数坐标") from exc
    if len(start) != 2 or len(goal) != 2:
        raise ValueError("坐标必须包含两个整数")
    for point in (start, goal):
        if not (0 <= point[0] < rows and 0 <= point[1] < columns):
            raise ValueError("坐标越界")
        if grid[point[0]][point[1]] == "#":
            raise ValueError("起终点不能是障碍")
    return grid, start, goal


def shortest_path(grid: list[str], start: tuple[int, int],
                  goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    rows, columns = len(grid), len(grid[0])
    queue = deque([start]); parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dr, dc in DIRECTIONS:
            neighbor = current[0] + dr, current[1] + dc
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns
                    and grid[neighbor[0]][neighbor[1]] == "." and neighbor not in parent):
                parent[neighbor] = current
                queue.append(neighbor)
    if goal not in parent:
        return None
    path = []; current: tuple[int, int] | None = goal
    while current is not None:
        path.append(current); current = parent[current]
    return path[::-1]


def save_result(path: Path, route: list[tuple[int, int]] | None) -> None:
    if route is None:
        content = "distance=-1\npath=NONE\n"
    else:
        coordinates = "->".join(f"({row},{column})" for row, column in route)
        content = f"distance={len(route) - 1}\npath={coordinates}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def run(input_path: Path, output_path: Path) -> None:
    grid, start, goal = load_problem(input_path)
    save_result(output_path, shortest_path(grid, start, goal))


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"第2题失败：{exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
