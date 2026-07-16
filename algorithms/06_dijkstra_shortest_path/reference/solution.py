"""参考实现：Dijkstra非负权最短路完整程序。"""

import argparse
import heapq
from pathlib import Path
import sys


WeightedAdjacency = list[list[tuple[int, int]]]


def parse_weighted_graph(text: str) -> tuple[WeightedAdjacency, int, int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    lines = text.strip().splitlines()
    first = lines[0].split()
    if len(first) != 4:
        raise ValueError("第一行必须是n m start goal")
    try:
        n, m, start, goal = map(int, first)
    except ValueError as error:
        raise ValueError("n、m、start和goal必须是整数") from error
    if n <= 0 or m < 0 or not 0 <= start < n or not 0 <= goal < n or len(lines) != m + 1:
        raise ValueError("图规模、起终点或边行数无效")
    adjacency: WeightedAdjacency = [[] for _ in range(n)]
    seen_edges: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"第{line_number}行必须是from to weight")
        try:
            source, destination, weight = map(int, parts)
        except ValueError as error:
            raise ValueError(f"第{line_number}行必须全部是整数") from error
        if not 0 <= source < n or not 0 <= destination < n or source == destination:
            raise ValueError("边端点越界或出现自环")
        if weight < 0:
            raise ValueError("Dijkstra不接受负权边")
        edge = (source, destination)
        if edge in seen_edges:
            raise ValueError(f"重复有向边: {edge}")
        seen_edges.add(edge)
        adjacency[source].append((destination, weight))
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency, start, goal


def _validate_adjacency(adjacency: WeightedAdjacency, start: int) -> None:
    if not isinstance(adjacency, list) or not adjacency or not 0 <= start < len(adjacency):
        raise ValueError("邻接表或起点无效")
    n = len(adjacency)
    for neighbors in adjacency:
        if not isinstance(neighbors, list):
            raise ValueError("每个邻接项必须是列表")
        for destination, weight in neighbors:
            if not 0 <= destination < n or not isinstance(weight, int) or weight < 0:
                raise ValueError("邻接边必须指向有效节点且权重为非负整数")


def dijkstra(
    adjacency: WeightedAdjacency, start: int
) -> tuple[list[int | None], list[int | None], int]:
    _validate_adjacency(adjacency, start)
    distances: list[int | None] = [None] * len(adjacency)
    parents: list[int | None] = [None] * len(adjacency)
    distances[start] = 0
    heap = [(0, start)]
    stale_entries = 0
    while heap:
        popped_distance, node = heapq.heappop(heap)
        if popped_distance != distances[node]:
            stale_entries += 1
            continue
        for neighbor, weight in adjacency[node]:
            candidate = popped_distance + weight
            if distances[neighbor] is None or candidate < distances[neighbor]:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(heap, (candidate, neighbor))
    return distances, parents, stale_entries


def reconstruct_path(
    parents: list[int | None], start: int, goal: int
) -> list[int]:
    if (
        not isinstance(parents, list)
        or not parents
        or not 0 <= start < len(parents)
        or not 0 <= goal < len(parents)
    ):
        raise ValueError("parents或起终点无效")
    if start == goal:
        return [start]
    path = []
    current: int | None = goal
    seen: set[int] = set()
    while current is not None and current not in seen:
        seen.add(current)
        path.append(current)
        if current == start:
            path.reverse()
            return path
        current = parents[current]
    return []


def solve(text: str) -> str:
    adjacency, start, goal = parse_weighted_graph(text)
    distances, parents, _ = dijkstra(adjacency, start)
    distance = distances[goal]
    path = [] if distance is None else reconstruct_path(parents, start, goal)
    distance_text = "-1" if distance is None else str(distance)
    path_text = "NONE" if not path else " ".join(map(str, path))
    return f"distance: {distance_text}\npath: {path_text}"


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dijkstra非负权最短路")
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
