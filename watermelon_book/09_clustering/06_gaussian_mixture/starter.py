"""学生练习：高斯混合聚类的EM。"""
import numpy as np
def gaussian_log_density(X: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray: raise NotImplementedError
def expectation_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray,float]: raise NotImplementedError
def maximization_step(X: np.ndarray, responsibilities: np.ndarray, *, covariance_regularization: float=1e-6) -> tuple[np.ndarray,np.ndarray,np.ndarray]: raise NotImplementedError
def fit_gaussian_mixture(X: np.ndarray, n_components: int, **options: object) -> dict[str,object]: raise NotImplementedError
