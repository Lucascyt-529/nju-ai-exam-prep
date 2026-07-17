"""学生练习：安全保存和恢复NumPy模型文件。"""

from pathlib import Path

import numpy as np


def save_array(path: Path, array: np.ndarray) -> None:
    raise NotImplementedError("请完成 save_array")


def load_array(path: Path, *, expected_ndim: int | None = None) -> np.ndarray:
    raise NotImplementedError("请完成 load_array")


def save_model_bundle(path: Path, mean: np.ndarray, scale: np.ndarray,
                      weights: np.ndarray, bias: float) -> None:
    raise NotImplementedError("请完成 save_model_bundle")


def load_model_bundle(path: Path) -> dict[str, np.ndarray | float]:
    raise NotImplementedError("请完成 load_model_bundle")


def predict_with_bundle(X: np.ndarray, bundle: dict[str, np.ndarray | float]) -> np.ndarray:
    raise NotImplementedError("请完成 predict_with_bundle")
