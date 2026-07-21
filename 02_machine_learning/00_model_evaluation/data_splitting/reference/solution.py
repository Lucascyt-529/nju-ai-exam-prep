"""参考适配层：用已验证的两份留出实现三份数据划分。"""
import importlib.util
from pathlib import Path
import numpy as np

def _load_core():
    root = Path(__file__).resolve().parents[4]
    path = root / "watermelon_book" / "02_model_evaluation_selection" / "01_data_splitting" / "reference" / "solution.py"
    spec = importlib.util.spec_from_file_location("data_splitting_core", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

_CORE = _load_core()
def train_validation_test_split_indices(n_samples, validation_size, test_size, seed):
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size 与 test_size 必须为正且总和小于1")
    remain, test = _CORE.train_test_split_indices(n_samples, test_size, seed)
    relative = validation_size / (1.0 - test_size)
    train_local, validation_local = _CORE.train_test_split_indices(len(remain), relative, seed + 1)
    return remain[train_local], remain[validation_local], test
def stratified_split_indices(y, validation_size, test_size, seed):
    labels = np.asarray(y)
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size 与 test_size 必须为正且总和小于1")
    remain, test = _CORE.stratified_train_test_split_indices(labels, test_size, seed)
    relative = validation_size / (1.0 - test_size)
    train_local, validation_local = _CORE.stratified_train_test_split_indices(labels[remain], relative, seed + 1)
    return remain[train_local], remain[validation_local], test
