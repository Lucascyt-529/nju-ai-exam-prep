"""学生练习：Mahalanobis距离与成对约束度量学习。"""
import numpy as np
def pairwise_mahalanobis(X: np.ndarray, M: np.ndarray) -> np.ndarray: raise NotImplementedError
def project_psd(M: np.ndarray) -> np.ndarray: raise NotImplementedError
def metric_objective(X: np.ndarray, y: np.ndarray, M: np.ndarray, *, margin: float=1.0, different_weight: float=1.0, regularization: float=0.01) -> float: raise NotImplementedError
def fit_pair_metric(X: np.ndarray, y: np.ndarray, **options: object) -> dict[str, object]: raise NotImplementedError
