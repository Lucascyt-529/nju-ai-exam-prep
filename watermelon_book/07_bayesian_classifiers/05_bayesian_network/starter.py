"""学生练习：二值贝叶斯网、学习、评分与推断。"""

import numpy as np


def build_binary_network(parents: dict[str, tuple[str, ...]], probability_one: dict[str, np.ndarray]) -> dict[str, object]:
    raise NotImplementedError("请完成 build_binary_network")


def topological_order(parents: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    raise NotImplementedError("请完成 topological_order")


def joint_probability(model: dict[str, object], assignment: dict[str, int]) -> float:
    raise NotImplementedError("请完成 joint_probability")


def query_posterior(model: dict[str, object], query: str, evidence: dict[str, int]) -> np.ndarray:
    raise NotImplementedError("请完成 query_posterior")


def fit_binary_cpts(data: np.ndarray, variable_names: tuple[str, ...],
                    parents: dict[str, tuple[str, ...]], *, alpha: float = 0.0) -> dict[str, object]:
    raise NotImplementedError("请完成 fit_binary_cpts")


def bic_score(model: dict[str, object], data: np.ndarray,
              variable_names: tuple[str, ...]) -> float:
    raise NotImplementedError("请完成 bic_score")


def greedy_ordered_structure(data: np.ndarray, variable_names: tuple[str, ...], *,
                             order: tuple[str, ...] | None = None,
                             max_parents: int = 2) -> dict[str, object]:
    raise NotImplementedError("请完成 greedy_ordered_structure")


def markov_blanket_probability_one(model: dict[str, object], variable: str,
                                   assignment: dict[str, int]) -> float:
    raise NotImplementedError("请完成 markov_blanket_probability_one")


def gibbs_query(model: dict[str, object], query: str, evidence: dict[str, int], *,
                n_samples: int = 5000, burn_in: int = 500,
                random_state: int = 0) -> dict[str, object]:
    raise NotImplementedError("请完成 gibbs_query")
