"""模拟卷2第2题参考实现：确定性Dijkstra与完整路径平局。"""

import argparse
import heapq
from pathlib import Path
import sys

import numpy as np


Graph = dict[str, list[tuple[str, float]]]


def load_problem(path: Path) -> tuple[Graph, str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError("输入为空")
    first = lines[0].split()
    if len(first) != 2:
        raise ValueError("第一行必须是n m")
    try:
        n, m = map(int, first)
    except ValueError as error:
        raise ValueError("n和m必须是整数") from error
    if n <= 0 or m < 0 or len(lines) != m + 2:
        raise ValueError("结点数、边数或输入行数错误")
    graph: Graph = {}
    edges: set[tuple[str, str]] = set()
    for line_number, line in enumerate(lines[1 : m + 1], start=2):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"第{line_number}行边格式错误")
        left, right, raw_weight = parts
        if left == right:
            raise ValueError("不允许自环")
        key = tuple(sorted((left, right)))
        if key in edges:
            raise ValueError("不允许重复无向边")
        edges.add(key)
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"第{line_number}行权重不是数值") from error
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("边权必须是非负有限数值")
        graph.setdefault(left, []).append((right, weight))
        graph.setdefault(right, []).append((left, weight))
    if len(graph) != n:
        raise ValueError("n必须等于边中出现的不同结点数")
    query = lines[-1].split()
    if len(query) != 2 or query[0] not in graph or query[1] not in graph:
        raise ValueError("查询起终点必须是图中结点")
    for node in graph:
        graph[node].sort(key=lambda item: item[0])
    return graph, query[0], query[1]


def shortest_path(graph: Graph, start: str, goal: str) -> tuple[float, list[str]] | None:
    if start not in graph or goal not in graph:
        raise ValueError("起终点必须在图中")
    best: dict[str, tuple[float, tuple[str, ...]]] = {start: (0.0, (start,))}
    heap: list[tuple[float, tuple[str, ...], str]] = [(0.0, (start,), start)]
    while heap:
        distance, path, node = heapq.heappop(heap)
        if best.get(node) != (distance, path):
            continue
        if node == goal:
            return distance, list(path)
        for neighbor, weight in graph[node]:
            candidate = (distance + weight, path + (neighbor,))
            current = best.get(neighbor)
            if current is None or candidate < current:
                best[neighbor] = candidate
                heapq.heappush(heap, (candidate[0], candidate[1], neighbor))
    return None


def run(input_path: Path, output_path: Path) -> None:
    graph, start, goal = load_problem(input_path)
    result = shortest_path(graph, start, goal)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if result is None:
        content = "distance=INF\npath=NONE\n"
    else:
        distance, path = result
        content = f"distance={distance:.6f}\npath={'->'.join(path)}\n"
    output_path.write_text(content, encoding="utf-8", newline="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.input, args.output)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
