import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = (
    ROOT
    / "03_data_processing"
    / "01_imputation_standardization"
    / "reference"
    / "solution.py"
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("data_processing_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_test_data_uses_training_fill_values_without_refitting() -> None:
    X_train = np.array([[1, 10], [3, np.nan], [5, 30]], dtype=float)
    X_test = np.array([[100, np.nan], [200, 999]], dtype=float)

    fill_values = solution.fit_mean_imputer(X_train)
    train_filled = solution.transform_mean_imputer(X_train, fill_values)
    test_filled = solution.transform_mean_imputer(X_test, fill_values)

    np.testing.assert_array_equal(fill_values, [3, 20])
    np.testing.assert_array_equal(train_filled, [[1, 10], [3, 20], [5, 30]])
    np.testing.assert_array_equal(test_filled, [[100, 20], [200, 999]])


def test_imputer_does_not_modify_input_array() -> None:
    X = np.array([[1, np.nan], [3, 5]], dtype=float)
    original = X.copy()

    result = solution.transform_mean_imputer(X, np.array([2, 5], dtype=float))

    assert np.isnan(result[0, 1]) is np.False_
    np.testing.assert_array_equal(X, original)


def test_all_missing_training_feature_is_rejected() -> None:
    X_train = np.array([[1, np.nan], [2, np.nan]], dtype=float)

    with pytest.raises(ValueError, match="全部缺失"):
        solution.fit_mean_imputer(X_train)


def test_standardizer_uses_training_statistics_for_test_data() -> None:
    X_train = np.array([[1, 10], [3, 20], [5, 30]], dtype=float)
    X_test = np.array([[100, 200], [200, 400]], dtype=float)

    means, scales = solution.fit_standardizer(X_train)
    train_scaled = solution.transform_standardizer(X_train, means, scales)
    test_scaled = solution.transform_standardizer(X_test, means, scales)

    np.testing.assert_allclose(train_scaled.mean(axis=0), [0, 0], atol=1e-12)
    np.testing.assert_allclose(train_scaled.std(axis=0), [1, 1], atol=1e-12)
    assert not np.allclose(test_scaled.mean(axis=0), [0, 0])
    np.testing.assert_array_equal(means, [3, 20])


def test_constant_feature_uses_unit_scale() -> None:
    X_train = np.array([[1, 5], [3, 5], [5, 5]], dtype=float)

    means, scales = solution.fit_standardizer(X_train)
    scaled = solution.transform_standardizer(X_train, means, scales)

    assert scales[1] == 1.0
    np.testing.assert_array_equal(scaled[:, 1], [0, 0, 0])


def test_transform_rejects_different_feature_count() -> None:
    X = np.array([[1, 2, 3]], dtype=float)

    with pytest.raises(ValueError, match="n_features"):
        solution.transform_standardizer(
            X,
            means=np.array([1, 2], dtype=float),
            scales=np.array([1, 1], dtype=float),
        )
