"""参考实现：一维DP最少硬币与组合恢复。"""

import argparse
from pathlib import Path
import sys


def parse_problem(text: str) -> tuple[list[int], int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    lines = text.strip().splitlines()
    if len(lines) != 2:
        raise ValueError("输入必须恰好包含两行")
    first = lines[0].split()
    if len(first) != 2:
        raise ValueError("第一行必须是k amount")
    try:
        k, amount = map(int, first)
        coins = [int(value) for value in lines[1].split()]
    except ValueError as error:
        raise ValueError("k、amount和coins必须是整数") from error
    if k <= 0 or amount < 0 or len(coins) != k:
        raise ValueError("面额数、目标金额或硬币数量无效")
    if any(coin <= 0 for coin in coins) or len(set(coins)) != len(coins):
        raise ValueError("面额必须是互不重复的正整数")
    return sorted(coins), amount


def minimum_coin_change(
    coins: list[int], amount: int
) -> tuple[int, list[int], list[int | None]]:
    if (
        not isinstance(coins, list)
        or not coins
        or any(not isinstance(coin, int) or isinstance(coin, bool) or coin <= 0 for coin in coins)
        or len(set(coins)) != len(coins)
        or not isinstance(amount, int)
        or isinstance(amount, bool)
        or amount < 0
    ):
        raise ValueError("coins必须是互异正整数列表且amount必须非负")
    ordered_coins = sorted(coins)
    dp: list[int | None] = [None] * (amount + 1)
    chosen_coin: list[int | None] = [None] * (amount + 1)
    dp[0] = 0
    for current in range(1, amount + 1):
        for coin in ordered_coins:
            if coin > current:
                break
            previous = dp[current - coin]
            if previous is None:
                continue
            candidate = previous + 1
            if (
                dp[current] is None
                or candidate < dp[current]
                or (candidate == dp[current] and coin < chosen_coin[current])
            ):
                dp[current] = candidate
                chosen_coin[current] = coin
    if dp[amount] is None:
        return -1, [], dp
    combination = []
    current = amount
    while current > 0:
        coin = chosen_coin[current]
        if coin is None:
            raise RuntimeError("DP恢复状态不一致")
        combination.append(coin)
        current -= coin
    combination.sort()
    return dp[amount], combination, dp


def solve(text: str) -> str:
    coins, amount = parse_problem(text)
    minimum, combination, _ = minimum_coin_change(coins, amount)
    coin_text = "NONE" if minimum == -1 else " ".join(map(str, combination))
    return f"minimum: {minimum}\ncoins: {coin_text}"


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="一维DP最少硬币")
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
