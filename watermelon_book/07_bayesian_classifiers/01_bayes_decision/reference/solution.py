"""参考实现：贝叶斯后验、条件风险与最小风险分类。"""

import numpy as np


def _finite_array(value: object) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and np.all(np.isfinite(value))


def _probability_vector(value: np.ndarray, name: str) -> None:
    if not _finite_array(value) or value.ndim != 1 or value.size < 2 or np.any(value < 0) or not np.isclose(np.sum(value), 1.0, atol=1e-10):
        raise ValueError(f"{name}必须是和为1的非负一维概率数组")


def _posterior_matrix(value: np.ndarray) -> None:
    if not _finite_array(value) or value.ndim != 2 or value.shape[0] == 0 or value.shape[1] < 2 or np.any(value < 0) or not np.allclose(np.sum(value, axis=1), 1.0, atol=1e-10):
        raise ValueError("posteriors必须是每行和为1的非负二维概率矩阵")


def posterior_from_likelihoods(priors: np.ndarray, class_likelihoods: np.ndarray) -> np.ndarray:
    _probability_vector(priors, "priors")
    if not _finite_array(class_likelihoods) or class_likelihoods.ndim != 2 or class_likelihoods.shape[0] == 0 or class_likelihoods.shape[1] != priors.size or np.any(class_likelihoods < 0):
        raise ValueError("class_likelihoods必须是形状(n,K)的非负有限数组")
    joint = class_likelihoods.astype(float, copy=False) * priors.astype(float, copy=False)[None, :]
    evidence = np.sum(joint, axis=1, keepdims=True)
    if np.any(evidence <= 0):
        raise ValueError("每个样本至少有一个类别的联合得分必须为正")
    return joint / evidence


def zero_one_loss_matrix(n_classes: int) -> np.ndarray:
    if not isinstance(n_classes, (int, np.integer)) or isinstance(n_classes, (bool, np.bool_)) or n_classes < 2:
        raise ValueError("n_classes必须是不小于2的整数")
    return np.ones((int(n_classes), int(n_classes))) - np.eye(int(n_classes))


def _validate_loss_matrix(loss_matrix: np.ndarray, n_classes: int) -> None:
    if not _finite_array(loss_matrix) or loss_matrix.shape != (n_classes, n_classes) or np.any(loss_matrix < 0):
        raise ValueError("loss_matrix必须是形状(K,K)的非负有限数组")


def conditional_risks(posteriors: np.ndarray, loss_matrix: np.ndarray) -> np.ndarray:
    _posterior_matrix(posteriors)
    _validate_loss_matrix(loss_matrix, posteriors.shape[1])
    return posteriors.astype(float, copy=False) @ loss_matrix.astype(float, copy=False).T


def minimum_risk_decisions(posteriors: np.ndarray, loss_matrix: np.ndarray) -> np.ndarray:
    return np.argmin(conditional_risks(posteriors, loss_matrix), axis=1)


def maximum_posterior_decisions(posteriors: np.ndarray) -> np.ndarray:
    _posterior_matrix(posteriors)
    return np.argmax(posteriors, axis=1)


def selected_risks(posteriors: np.ndarray, loss_matrix: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    risks = conditional_risks(posteriors, loss_matrix)
    if not isinstance(decisions, np.ndarray) or decisions.shape != (posteriors.shape[0],) or not np.issubdtype(decisions.dtype, np.integer) or np.any(decisions < 0) or np.any(decisions >= posteriors.shape[1]):
        raise ValueError("decisions必须是形状(n,)的合法整数动作下标")
    return risks[np.arange(risks.shape[0]), decisions]


def binary_positive_threshold(false_positive_cost: float, false_negative_cost: float) -> float:
    values = (false_positive_cost, false_negative_cost)
    if any(not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("两种错误代价必须是正有限数值")
    return float(false_positive_cost / (false_positive_cost + false_negative_cost))
