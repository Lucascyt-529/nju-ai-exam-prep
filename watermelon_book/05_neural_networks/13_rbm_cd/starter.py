"""学生练习：RBM条件概率与CD-1。"""
import numpy as np
def hidden_probabilities(visible: np.ndarray, weights: np.ndarray, hidden_bias: np.ndarray) -> np.ndarray: raise NotImplementedError
def visible_probabilities(hidden: np.ndarray, weights: np.ndarray, visible_bias: np.ndarray) -> np.ndarray: raise NotImplementedError
def cd1_step(visible_data: np.ndarray, weights: np.ndarray, visible_bias: np.ndarray, hidden_bias: np.ndarray, *, learning_rate: float=.1, random_state: int=0) -> dict[str,object]: raise NotImplementedError
def fit_rbm_cd1(visible_data: np.ndarray, n_hidden: int, **options: object) -> dict[str,object]: raise NotImplementedError
