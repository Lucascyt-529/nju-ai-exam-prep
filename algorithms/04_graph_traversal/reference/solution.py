"""参考实现：无向图BFS、DFS与连通分量完整程序。"""

import argparse
from collections import deque
from pathlib import Path
import sys


def parse_graph(text: str) -> tuple[list[list[int]], int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    lines = text.strip().splitlines()
    first = lines[0].split()
    if len(first) != 3:
        raise ValueError("第一行必须是n m start")
    try:
        n, m, start = map(int, first)
    except ValueError as error:
        raise ValueError("n、m和start必须是整数") from error
    if n <= 0 or m < 0 or not 0 <= start < n or len(lines) != m + 1:
        raise ValueError("图规模、起点或边行数无效")
    adjacency = [[] for _ in range(n)]
    seen_edges: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"第{line_number}行必须包含两个端点")
        try:
            left, right = map(int, parts)
        except ValueError as error:
            raise ValueError(f"第{line_number}行端点必须是整数") from error
        if not 0 <= left < n or not 0 <= right < n or left == right:
            raise ValueError("边端点越界或出现自环")
        edge = (min(left, right), max(left, right))
        if edge in seen_edges:
            raise ValueError(f"重复无向边: {edge}")
        seen_edges.add(edge)
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency, start


def _validate_start(adjacency: list[list[int]], start: int) -> None:
    if not isinstance(adjacency, list) or not adjacency or not 0 <= start < len(adjacency):
        raise ValueError("邻接表或起点无效")


def bfs(adjacency: list[list[int]], start: int) -> list[int]:
    _validate_start(adjacency, start)
    visited = [False] * len(adjacency)
    visited[start] = True
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return order


def dfs(adjacency: list[list[int]], start: int) -> list[int]:
    _validate_start(adjacency, start)
    visited = [False] * len(adjacency)
    visited[start] = True
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in reversed(adjacency[node]):
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return order


def connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    if not isinstance(adjacency, list) or not adjacency:
        raise ValueError("adjacency必须是非空邻接表")
    visited = [False] * len(adjacency)
    components = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def solve(text: str) -> str:
    adjacency, start = parse_graph(text)
    lines = [
        "bfs: " + " ".join(map(str, bfs(adjacency, start))),
        "dfs: " + " ".join(map(str, dfs(adjacency, start))),
        "components:",
    ]
    lines.extend(" ".join(map(str, component)) for component in connected_components(adjacency))
    return "\n".join(lines)


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="无向图BFS、DFS和连通分量")
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
