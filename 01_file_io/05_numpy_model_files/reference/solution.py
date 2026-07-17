"""参考实现：安全保存和恢复NumPy数组与模型参数包。"""

from pathlib import Path
from numbers import Real

import numpy as np


MODEL_KEYS = {"mean", "scale", "weights", "bias"}


def _finite_array(value: np.ndarray, name: str, *, ndim: int) -> None:
    if (not isinstance(value, np.ndarray) or value.ndim != ndim or 0 in value.shape
            or not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value))):
        raise ValueError(f"{name}必须是非空有限数值{ndim}维数组")


def save_array(path: Path, array: np.ndarray) -> None:
    """保存单个数组，并严格使用调用者给出的路径。"""
    _finite_array(array, "array", ndim=array.ndim if isinstance(array, np.ndarray) else 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.save(file, array, allow_pickle=False)


def load_array(path: Path, *, expected_ndim: int | None = None) -> np.ndarray:
    with path.open("rb") as file:
        array = np.load(file, allow_pickle=False)
    if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError("数组文件必须包含有限数值数组")
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(f"数组维数应为{expected_ndim}，实际为{array.ndim}")
    return array.copy()


def _validate_model(mean: np.ndarray, scale: np.ndarray, weights: np.ndarray,
                    bias: float) -> None:
    _finite_array(mean, "mean", ndim=1)
    _finite_array(scale, "scale", ndim=1)
    _finite_array(weights, "weights", ndim=1)
    if mean.shape != scale.shape or mean.shape != weights.shape:
        raise ValueError("mean、scale和weights必须具有相同形状(n_features,)")
    if np.any(scale <= 0):
        raise ValueError("scale必须全部大于0")
    if (isinstance(bias, (bool, np.bool_)) or not isinstance(bias, Real)
            or not np.isfinite(bias)):
        raise ValueError("bias必须是有限数值标量")


def save_model_bundle(path: Path, mean: np.ndarray, scale: np.ndarray,
                      weights: np.ndarray, bias: float) -> None:
    """保存预处理参数与线性模型参数，避免只保存weights导致无法复现。"""
    _validate_model(mean, scale, weights, bias)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(
            file,
            mean=mean.astype(float, copy=False),
            scale=scale.astype(float, copy=False),
            weights=weights.astype(float, copy=False),
            bias=np.asarray(float(bias)),
        )


def load_model_bundle(path: Path) -> dict[str, np.ndarray | float]:
    """只接受预期键、形状和有限数值；不允许pickle对象。"""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != MODEL_KEYS:
            raise ValueError(f"模型文件必须包含且只包含: {sorted(MODEL_KEYS)}")
        mean = archive["mean"].astype(float, copy=True)
        scale = archive["scale"].astype(float, copy=True)
        weights = archive["weights"].astype(float, copy=True)
        bias_array = archive["bias"].astype(float, copy=True)
    if bias_array.shape != ():
        raise ValueError("bias在文件中必须是0维标量数组")
    bias = float(bias_array)
    _validate_model(mean, scale, weights, bias)
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def predict_with_bundle(X: np.ndarray, bundle: dict[str, np.ndarray | float]) -> np.ndarray:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or 0 in X.shape
            or not np.issubdtype(X.dtype, np.number) or not np.all(np.isfinite(X))):
        raise ValueError("X必须是非空有限数值二维数组")
    if not isinstance(bundle, dict) or set(bundle) != MODEL_KEYS:
        raise ValueError("bundle键不完整")
    mean = np.asarray(bundle["mean"]); scale = np.asarray(bundle["scale"])
    weights = np.asarray(bundle["weights"]); bias = bundle["bias"]
    _validate_model(mean, scale, weights, bias)
    if X.shape[1] != len(mean):
        raise ValueError("X特征数与模型不一致")
    return ((X.astype(float) - mean) / scale) @ weights + float(bias)
