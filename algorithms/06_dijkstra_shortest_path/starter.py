"""学生练习：Dijkstra非负权最短路完整程序。"""


WeightedAdjacency = list[list[tuple[int, int]]]


def parse_weighted_graph(text: str) -> tuple[WeightedAdjacency, int, int]:
    raise NotImplementedError("请完成 parse_weighted_graph")


def dijkstra(
    adjacency: WeightedAdjacency, start: int
) -> tuple[list[int | None], list[int | None], int]:
    raise NotImplementedError("请完成 dijkstra")


def reconstruct_path(
    parents: list[int | None], start: int, goal: int
) -> list[int]:
    raise NotImplementedError("请完成 reconstruct_path")


def solve(text: str) -> str:
    raise NotImplementedError("请完成 solve")
