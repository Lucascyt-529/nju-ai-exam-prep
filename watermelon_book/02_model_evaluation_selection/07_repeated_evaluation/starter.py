"""学生练习：重复留出与重复交叉验证。"""

from collections.abc import Callable

import numpy as np


def generate_repetition_seeds(repetitions: int, base_seed: int) -> np.ndarray:
    """从基础种子生成每次重复使用的独立、可复现子种子。"""
    raise NotImplementedError("请完成 generate_repetition_seeds")


def repeated_holdout_scores(
    X: np.ndarray,
    y: np.ndarray,
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    repetitions: int,
    test_size: float,
    base_seed: int,
    stratified: bool = False,
) -> dict[str, object]:
    """重复随机留出并返回分数、种子及每次训练/测试索引。"""
    raise NotImplementedError("请完成 repeated_holdout_scores")


def repeated_kfold_scores(
    X: np.ndarray,
    y: np.ndarray,
    fit_model: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    repetitions: int,
    n_splits: int,
    base_seed: int,
    stratified: bool = False,
) -> dict[str, object]:
    """重复K折评估并返回(p, k)分数矩阵、种子及各套折索引。"""
    raise NotImplementedError("请完成 repeated_kfold_scores")


def summarize_repeated_scores(scores: np.ndarray) -> dict[str, object]:
    """汇总运行均值、运行波动、每次重复均值及实际拟合次数。"""
    raise NotImplementedError("请完成 summarize_repeated_scores")


def evaluation_run_count(
    repetitions: int, *, n_splits: int | None = None
) -> int:
    """返回重复留出或重复K折所需的模型拟合次数。"""
    raise NotImplementedError("请完成 evaluation_run_count")
