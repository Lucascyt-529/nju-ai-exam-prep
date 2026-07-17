"""参考实现：0/1背包、二维DP与物品恢复。"""

import argparse
from pathlib import Path
import sys


def _validate_inputs(weights: list[int], values: list[int], capacity: int) -> None:
    valid_weights = (
        isinstance(weights, list)
        and all(
            isinstance(weight, int) and not isinstance(weight, bool) and weight > 0
            for weight in weights
        )
    )
    valid_values = (
        isinstance(values, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        )
    )
    valid_capacity = (
        isinstance(capacity, int) and not isinstance(capacity, bool) and capacity >= 0
    )
    if not valid_weights or not valid_values or len(weights) != len(values) or not valid_capacity:
        raise ValueError("weights必须是正整数列表，values和capacity必须为非负整数")


def parse_problem(text: str) -> tuple[list[int], list[int], int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.strip().split("\n")
    first = lines[0].split()
    if len(first) != 2:
        raise ValueError("第一行必须是n capacity")
    try:
        item_count, capacity = map(int, first)
    except ValueError as error:
        raise ValueError("n和capacity必须是整数") from error
    if item_count < 0 or capacity < 0:
        raise ValueError("n和capacity必须为非负整数")
    if len(lines) != item_count + 1:
        raise ValueError("物品行数必须与n一致")

    weights: list[int] = []
    values: list[int] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("每个物品行必须是weight value")
        try:
            weight, value = map(int, fields)
        except ValueError as error:
            raise ValueError("weight和value必须是整数") from error
        weights.append(weight)
        values.append(value)
    _validate_inputs(weights, values, capacity)
    return weights, values, capacity


def build_knapsack_table(
    weights: list[int], values: list[int], capacity: int
) -> list[list[int]]:
    _validate_inputs(weights, values, capacity)
    table = [[0] * (capacity + 1) for _ in range(len(weights) + 1)]
    for item_number, (weight, value) in enumerate(zip(weights, values), start=1):
        for current_capacity in range(capacity + 1):
            exclude_value = table[item_number - 1][current_capacity]
            table[item_number][current_capacity] = exclude_value
            if weight <= current_capacity:
                include_value = table[item_number - 1][current_capacity - weight] + value
                if include_value > exclude_value:
                    table[item_number][current_capacity] = include_value
    return table


def _validate_table(
    weights: list[int], values: list[int], capacity: int, table: list[list[int]]
) -> None:
    if (
        not isinstance(table, list)
        or len(table) != len(weights) + 1
        or any(not isinstance(row, list) or len(row) != capacity + 1 for row in table)
        or any(
            not isinstance(cell, int) or isinstance(cell, bool) or cell < 0
            for row in table
            for cell in row
        )
    ):
        raise ValueError("table形状或元素类型无效")
    expected = build_knapsack_table(weights, values, capacity)
    if table != expected:
        raise ValueError("table不符合0/1背包状态转移")


def reconstruct_selected_items(
    weights: list[int], values: list[int], capacity: int, table: list[list[int]]
) -> list[int]:
    _validate_inputs(weights, values, capacity)
    _validate_table(weights, values, capacity, table)
    selected_reversed: list[int] = []
    current_capacity = capacity
    for item_number in range(len(weights), 0, -1):
        if table[item_number][current_capacity] != table[item_number - 1][current_capacity]:
            item_index = item_number - 1
            selected_reversed.append(item_index)
            current_capacity -= weights[item_index]
    return list(reversed(selected_reversed))


def zero_one_knapsack(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int], list[list[int]]]:
    table = build_knapsack_table(weights, values, capacity)
    selected = reconstruct_selected_items(weights, values, capacity, table)
    return table[-1][-1], selected, table


def solve(text: str) -> str:
    weights, values, capacity = parse_problem(text)
    maximum, selected, _ = zero_one_knapsack(weights, values, capacity)
    total_weight = sum(weights[index] for index in selected)
    item_text = "NONE" if not selected else " ".join(map(str, selected))
    return f"maximum: {maximum}\nweight: {total_weight}\nitems: {item_text}"


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="0/1背包与物品恢复")
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
