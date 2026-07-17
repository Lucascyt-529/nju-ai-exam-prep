"""学生练习：二项混合模型的EM。"""
import numpy as np
def expectation_step(heads: np.ndarray, tosses: np.ndarray, mixing: np.ndarray, probabilities: np.ndarray) -> np.ndarray: raise NotImplementedError
def maximization_step(heads: np.ndarray, tosses: np.ndarray, responsibilities: np.ndarray) -> tuple[np.ndarray,np.ndarray]: raise NotImplementedError
def observed_log_likelihood(heads: np.ndarray, tosses: np.ndarray, mixing: np.ndarray, probabilities: np.ndarray) -> float: raise NotImplementedError
def expected_complete_log_likelihood(heads: np.ndarray, tosses: np.ndarray, responsibilities: np.ndarray, mixing: np.ndarray, probabilities: np.ndarray) -> float: raise NotImplementedError
def responsibility_entropy(responsibilities: np.ndarray) -> float: raise NotImplementedError
def evidence_lower_bound(heads: np.ndarray, tosses: np.ndarray, responsibilities: np.ndarray, mixing: np.ndarray, probabilities: np.ndarray) -> float: raise NotImplementedError
def posterior_kl_gap(heads: np.ndarray, tosses: np.ndarray, responsibilities: np.ndarray, mixing: np.ndarray, probabilities: np.ndarray) -> float: raise NotImplementedError
def em_step_diagnostics(heads: np.ndarray, tosses: np.ndarray, mixing: np.ndarray, probabilities: np.ndarray) -> dict[str,object]: raise NotImplementedError
def fit_coin_mixture_em(heads: np.ndarray, tosses: np.ndarray, **options: object) -> dict[str,object]: raise NotImplementedError
