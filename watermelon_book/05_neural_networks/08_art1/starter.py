"""学生练习：二值ART1选择、警戒、共振与增量训练。"""

import numpy as np


def choice_scores(pattern: np.ndarray, prototypes: np.ndarray, *, alpha: float) -> np.ndarray:
    raise NotImplementedError("请完成 choice_scores")


def match_ratio(pattern: np.ndarray, prototype: np.ndarray) -> float:
    raise NotImplementedError("请完成 match_ratio")


def select_resonant_category(
    pattern: np.ndarray,
    prototypes: np.ndarray,
    *,
    alpha: float,
    vigilance: float,
) -> int | None:
    raise NotImplementedError("请完成 select_resonant_category")


def train_art1(
    X: np.ndarray,
    *,
    alpha: float = 0.001,
    vigilance: float = 0.8,
    max_categories: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    raise NotImplementedError("请完成 train_art1")


def predict_art1(
    X: np.ndarray,
    prototypes: np.ndarray,
    *,
    alpha: float = 0.001,
    vigilance: float = 0.8,
) -> np.ndarray:
    raise NotImplementedError("请完成 predict_art1")
