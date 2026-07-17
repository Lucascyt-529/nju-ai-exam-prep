"""学生练习：0/1背包、二维DP与物品恢复。"""


def parse_problem(text: str) -> tuple[list[int], list[int], int]:
    raise NotImplementedError("请完成 parse_problem")


def build_knapsack_table(
    weights: list[int], values: list[int], capacity: int
) -> list[list[int]]:
    raise NotImplementedError("请完成 build_knapsack_table")


def reconstruct_selected_items(
    weights: list[int], values: list[int], capacity: int, table: list[list[int]]
) -> list[int]:
    raise NotImplementedError("请完成 reconstruct_selected_items")


def zero_one_knapsack(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int], list[list[int]]]:
    raise NotImplementedError("请完成 zero_one_knapsack")


def solve(text: str) -> str:
    raise NotImplementedError("请完成 solve")

