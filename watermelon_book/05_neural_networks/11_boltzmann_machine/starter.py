"""学生练习：Boltzmann能量、概率和Gibbs采样。"""
import numpy as np
def enumerate_binary_states(n_units: int) -> np.ndarray: raise NotImplementedError
def energy(states: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray: raise NotImplementedError
def exact_distribution(weights: np.ndarray, biases: np.ndarray, *, temperature: float=1.0) -> dict[str, np.ndarray]: raise NotImplementedError
def conditional_probability_one(state: np.ndarray, unit: int, weights: np.ndarray, biases: np.ndarray, *, temperature: float=1.0) -> float: raise NotImplementedError
def gibbs_sample(initial_state: np.ndarray, weights: np.ndarray, biases: np.ndarray, *, n_sweeps: int, random_state: int=0, temperature: float=1.0) -> np.ndarray: raise NotImplementedError
