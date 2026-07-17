"""学生练习：LLE邻居、重构权重和谱嵌入。"""
import numpy as np
def nearest_neighbors(X: np.ndarray, n_neighbors: int) -> np.ndarray: raise NotImplementedError
def reconstruction_weights(X: np.ndarray, neighbors: np.ndarray, *, regularization: float=1e-3) -> np.ndarray: raise NotImplementedError
def fit_lle(X: np.ndarray, n_neighbors: int, n_components: int, *, regularization: float=1e-3) -> dict[str,np.ndarray]: raise NotImplementedError
