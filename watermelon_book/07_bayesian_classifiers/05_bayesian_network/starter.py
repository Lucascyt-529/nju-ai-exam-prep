"""学生练习：二值贝叶斯网与枚举精确推断。"""

import numpy as np


def build_binary_network(parents: dict[str, tuple[str, ...]], probability_one: dict[str, np.ndarray]) -> dict[str, object]:
    raise NotImplementedError("请完成 build_binary_network")


def topological_order(parents: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    raise NotImplementedError("请完成 topological_order")


def joint_probability(model: dict[str, object], assignment: dict[str, int]) -> float:
    raise NotImplementedError("请完成 joint_probability")


def query_posterior(model: dict[str, object], query: str, evidence: dict[str, int]) -> np.ndarray:
    raise NotImplementedError("请完成 query_posterior")
