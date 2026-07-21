"""学生练习：感知机线性得分、预测和逐样本训练。"""

import numpy as np


def decision_function(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    raise NotImplementedError("请完成 decision_function")


def predict_perceptron(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    raise NotImplementedError("请完成 predict_perceptron")


def train_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 1.0,
    max_epochs: int = 100,
) -> tuple[np.ndarray, float, list[int]]:
    raise NotImplementedError("请完成 train_perceptron")
