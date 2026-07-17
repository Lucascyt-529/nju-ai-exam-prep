"""学生练习：离散与高斯分布的极大似然估计。"""

import numpy as np


def class_prior_mle(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 class_prior_mle")


def bernoulli_mle(samples: np.ndarray) -> float:
    raise NotImplementedError("请完成 bernoulli_mle")


def categorical_probabilities(
    samples: np.ndarray, categories: np.ndarray, *, alpha: float = 0.0
) -> np.ndarray:
    raise NotImplementedError("请完成 categorical_probabilities")


def gaussian_mle(samples: np.ndarray) -> tuple[float, float]:
    raise NotImplementedError("请完成 gaussian_mle")


def multivariate_gaussian_mle(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 multivariate_gaussian_mle")
