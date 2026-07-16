"""学生练习：二维DP最长公共子序列。"""


def parse_sequences(text: str) -> tuple[str, str]:
    raise NotImplementedError("请完成 parse_sequences")


def lcs_table(first: str, second: str) -> list[list[int]]:
    raise NotImplementedError("请完成 lcs_table")


def reconstruct_lcs(first: str, second: str, table: list[list[int]]) -> str:
    raise NotImplementedError("请完成 reconstruct_lcs")


def solve(text: str) -> str:
    raise NotImplementedError("请完成 solve")
