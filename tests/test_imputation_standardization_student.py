import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARTER = (
    ROOT / "03_data_processing" / "01_imputation_standardization" / "starter.py"
)


def load_student_module():
    spec = importlib.util.spec_from_file_location("preprocessing_student", STARTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


student = load_student_module()


def test_train_and_test_use_training_fill_values() -> None:
    X_train = np.array([[1, 10], [3, np.nan], [5, 30]], dtype=float)
    X_test = np.array([[100, np.nan], [200, 999]], dtype=float)

    fill_values = student.fit_mean_imputer(X_train)
    train_filled = student.transform_mean_imputer(X_train, fill_values)
    test_filled = student.transform_mean_imputer(X_test, fill_values)

    np.testing.assert_array_equal(fill_values, [3, 20])
    np.testing.assert_array_equal(train_filled, [[1, 10], [3, 20], [5, 30]])
    np.testing.assert_array_equal(test_filled, [[100, 20], [200, 999]])


def test_imputation_does_not_modify_inputs() -> None:
    X = np.array([[1, np.nan], [3, 5]], dtype=float)
    original = X.copy()

    student.transform_mean_imputer(X, np.array([2, 5], dtype=float))

    np.testing.assert_array_equal(X, original)


def test_standardizer_fits_each_feature_separately() -> None:
    X_train = np.array([[1, 10], [3, 20], [5, 30]], dtype=float)

    means, scales = student.fit_standardizer(X_train)
    scaled = student.transform_standardizer(X_train, means, scales)

    np.testing.assert_array_equal(means, [3, 20])
    np.testing.assert_allclose(scaled.mean(axis=0), [0, 0], atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), [1, 1], atol=1e-12)
    assert means.shape == scales.shape == (2,)


def test_test_data_reuses_training_standardization_parameters() -> None:
    X_train = np.array([[1, 10], [3, 20], [5, 30]], dtype=float)
    X_test = np.array([[100, 200], [200, 400]], dtype=float)

    means, scales = student.fit_standardizer(X_train)
    test_scaled = student.transform_standardizer(X_test, means, scales)

    assert not np.allclose(test_scaled.mean(axis=0), [0, 0])


def test_constant_feature_uses_unit_scale() -> None:
    X_train = np.array([[1, 5], [3, 5], [5, 5]], dtype=float)

    means, scales = student.fit_standardizer(X_train)
    scaled = student.transform_standardizer(X_train, means, scales)

    assert scales[1] == 1.0
    np.testing.assert_array_equal(scaled[:, 1], [0, 0, 0])

