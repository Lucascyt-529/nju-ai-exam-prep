"""学生练习：后验概率、条件风险与最小风险决策。"""

import numpy as np


def posterior_from_likelihoods(
    priors: np.ndarray, class_likelihoods: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("请完成 posterior_from_likelihoods")


def zero_one_loss_matrix(n_classes: int) -> np.ndarray:
    raise NotImplementedError("请完成 zero_one_loss_matrix")


def conditional_risks(
    posteriors: np.ndarray, loss_matrix: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("请完成 conditional_risks")


def minimum_risk_decisions(
    posteriors: np.ndarray, loss_matrix: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("请完成 minimum_risk_decisions")
