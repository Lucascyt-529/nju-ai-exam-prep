import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "02_numpy_basics" / "00_array_creation_dtypes" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("numpy_creation_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_array_creation_and_description() -> None:
    vector = solution.make_float_vector([1, 2.5, 3])
    zeros = solution.make_zero_matrix(2, 3)

    np.testing.assert_allclose(vector, [1.0, 2.5, 3.0])
    assert vector.shape == (3,)
    assert vector.dtype == np.float64
    assert zeros.shape == (2, 3)
    assert zeros.dtype == np.float64
    assert np.count_nonzero(zeros) == 0
    assert solution.describe_array(zeros) == {
        "shape": (2, 3),
        "ndim": 2,
        "size": 6,
        "dtype": "float64",
    }


def test_step_sequence_is_left_closed_right_open() -> None:
    result = solution.make_step_sequence(1, 8, 2)
    np.testing.assert_allclose(result, [1.0, 3.0, 5.0, 7.0])


def test_float_conversion_returns_independent_array() -> None:
    original = np.array([1, 2, 3], dtype=int)
    converted = solution.convert_to_float(original)
    converted[0] = 99.0

    assert converted.dtype == np.float64
    assert original.tolist() == [1, 2, 3]


@pytest.mark.parametrize("values", [[], [[1, 2]], [1, np.nan]])
def test_invalid_vectors_are_rejected(values) -> None:
    with pytest.raises(ValueError):
        solution.make_float_vector(values)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (-1, 2)])
def test_invalid_zero_matrix_shapes_are_rejected(shape) -> None:
    with pytest.raises(ValueError):
        solution.make_zero_matrix(*shape)


def test_zero_step_is_rejected() -> None:
    with pytest.raises(ValueError):
        solution.make_step_sequence(0, 1, 0)
