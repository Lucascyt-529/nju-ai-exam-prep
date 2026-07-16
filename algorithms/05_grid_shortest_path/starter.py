"""学生练习：网格BFS最短路与路径恢复。"""


Cell = tuple[int, int]


def parse_grid(text: str) -> tuple[list[str], Cell, Cell]:
    raise NotImplementedError("请完成 parse_grid")


def shortest_grid_path(grid: list[str], start: Cell, goal: Cell) -> tuple[int, list[Cell]]:
    raise NotImplementedError("请完成 shortest_grid_path")


def solve(text: str) -> str:
    raise NotImplementedError("请完成 solve")
