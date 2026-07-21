import importlib.util
from pathlib import Path
import numpy as np

def test_three_way_split():
    path = Path(__file__).resolve().parents[1] / "reference" / "solution.py"
    spec = importlib.util.spec_from_file_location("split_reference", path)
    solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
    first = solution.train_validation_test_split_indices(20, 0.2, 0.2, 4)
    second = solution.train_validation_test_split_indices(20, 0.2, 0.2, 4)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(np.unique(np.concatenate(first))) == 20
    y = np.array([0] * 10 + [1] * 10)
    assert all(set(y[index]) == {0, 1} for index in solution.stratified_split_indices(y, 0.2, 0.2, 4))
