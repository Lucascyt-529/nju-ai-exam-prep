import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "03_data_processing" / "04_outlier_handling"
SOLUTION = TOPIC / "reference" / "solution.py"
STARTER = TOPIC / "starter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(SOLUTION, "outlier_solution")
starter = load_module(STARTER, "outlier_starter")


def training_data() -> np.ndarray:
    return np.array(
        [
            [1.0, 10.0, 5.0],
            [2.0, 11.0, 5.0],
            [3.0, 12.0, 5.0],
            [4.0, 13.0, 5.0],
            [100.0, 14.0, 5.0],
        ]
    )


def test_iqr_bounds_are_computed_columnwise_from_training_data() -> None:
    lower, upper = solution.fit_iqr_bounds(training_data())
    assert lower.shape == upper.shape == (3,)
    np.testing.assert_array_equal(lower, [-1.0, 8.0, 5.0])
    np.testing.assert_array_equal(upper, [7.0, 16.0, 5.0])


def test_custom_multiplier_changes_whisker_width() -> None:
    X = np.arange(5.0).reshape(-1, 1)
    lower, upper = solution.fit_iqr_bounds(X, multiplier=2.0)
    np.testing.assert_array_equal(lower, [-3.0])
    np.testing.assert_array_equal(upper, [7.0])


def test_detection_broadcasts_feature_bounds_and_keeps_matrix_shape() -> None:
    X = training_data()
    lower, upper = solution.fit_iqr_bounds(X)
    mask = solution.detect_outliers(X, lower, upper)
    assert mask.dtype == np.bool_ and mask.shape == X.shape
    np.testing.assert_array_equal(mask[:, 0], [False, False, False, False, True])
    assert not np.any(mask[:, 1:])


def test_values_equal_to_bounds_are_not_outliers() -> None:
    X = np.array([[-1.0, 16.0], [7.0, 8.0]])
    mask = solution.detect_outliers(
        X, np.array([-1.0, 8.0]), np.array([7.0, 16.0])
    )
    assert not np.any(mask)


def test_test_data_uses_training_bounds_without_refitting() -> None:
    X_train = np.arange(5.0).reshape(-1, 1)
    X_test = np.array([[100.0], [200.0], [300.0]])
    lower, upper = solution.fit_iqr_bounds(X_train)
    correct_mask = solution.detect_outliers(X_test, lower, upper)
    wrong_lower, wrong_upper = solution.fit_iqr_bounds(X_test)
    leaked_mask = solution.detect_outliers(X_test, wrong_lower, wrong_upper)
    np.testing.assert_array_equal(correct_mask[:, 0], [True, True, True])
    assert not np.any(leaked_mask)


def test_zero_iqr_training_column_flags_new_test_value() -> None:
    X_train = np.full((4, 1), 5.0)
    lower, upper = solution.fit_iqr_bounds(X_train)
    mask = solution.detect_outliers(np.array([[5.0], [6.0]]), lower, upper)
    np.testing.assert_array_equal(lower, [5.0])
    np.testing.assert_array_equal(upper, [5.0])
    np.testing.assert_array_equal(mask[:, 0], [False, True])


def test_summary_counts_features_and_rows_separately() -> None:
    mask = np.array(
        [[True, False, False], [False, True, True], [False, False, False]]
    )
    feature_counts, row_flags, row_count = solution.summarize_outliers(mask)
    np.testing.assert_array_equal(feature_counts, [1, 1, 1])
    np.testing.assert_array_equal(row_flags, [True, True, False])
    assert row_count == 2


def test_clipping_uses_per_feature_bounds() -> None:
    X = np.array([[-5.0, 20.0], [10.0, 0.0], [3.0, 12.0]])
    clipped = solution.clip_outliers(
        X, np.array([-1.0, 8.0]), np.array([7.0, 16.0])
    )
    np.testing.assert_array_equal(clipped, [[-1.0, 16.0], [7.0, 8.0], [3.0, 12.0]])


def test_clipping_does_not_modify_input_or_share_memory() -> None:
    X = np.array([[-5.0], [2.0], [10.0]])
    original = X.copy()
    clipped = solution.clip_outliers(X, np.array([0.0]), np.array([5.0]))
    np.testing.assert_array_equal(X, original)
    assert not np.shares_memory(X, clipped)


def test_robust_scaler_uses_column_median_and_iqr() -> None:
    medians, scales = solution.fit_robust_scaler(training_data())
    np.testing.assert_array_equal(medians, [3.0, 12.0, 5.0])
    np.testing.assert_array_equal(scales, [2.0, 2.0, 1.0])


def test_robust_transform_uses_training_state_on_test_data() -> None:
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    X_test = np.array([[200.0], [300.0], [400.0]])
    medians, scales = solution.fit_robust_scaler(X_train)
    transformed = solution.transform_robust_scaler(X_test, medians, scales)
    np.testing.assert_array_equal(medians, [2.0])
    np.testing.assert_array_equal(scales, [2.0])
    np.testing.assert_array_equal(transformed[:, 0], [99.0, 149.0, 199.0])
    assert not np.isclose(np.median(transformed), 0.0)


def test_constant_column_uses_safe_unit_scale() -> None:
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    medians, scales = solution.fit_robust_scaler(X)
    transformed = solution.transform_robust_scaler(X, medians, scales)
    assert scales[1] == 1.0
    np.testing.assert_array_equal(transformed[:, 1], [0.0, 0.0, 0.0])


def test_robust_transform_does_not_modify_input() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = X.copy()
    solution.transform_robust_scaler(
        X, np.array([2.0, 3.0]), np.array([1.0, 1.0])
    )
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize(
    "X",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 2)),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([[True, False]]),
        [[1.0, 2.0]],
    ],
)
def test_invalid_training_matrices_are_rejected(X) -> None:
    with pytest.raises(ValueError):
        solution.fit_iqr_bounds(X)


@pytest.mark.parametrize("multiplier", [0, -1, np.nan, np.inf, True])
def test_invalid_iqr_multipliers_are_rejected(multiplier) -> None:
    with pytest.raises(ValueError):
        solution.fit_iqr_bounds(training_data(), multiplier=multiplier)


def test_invalid_bound_shapes_and_order_are_rejected() -> None:
    X = np.ones((2, 2))
    with pytest.raises(ValueError, match="n_features"):
        solution.detect_outliers(X, np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="不能大于"):
        solution.clip_outliers(X, np.array([2.0, 0.0]), np.array([1.0, 1.0]))


@pytest.mark.parametrize(
    "mask",
    [
        np.array([True, False]),
        np.array([[0, 1]]),
        np.empty((0, 2), dtype=bool),
    ],
)
def test_invalid_summary_masks_are_rejected(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solution.summarize_outliers(mask)


def test_invalid_robust_scaler_state_is_rejected() -> None:
    X = np.ones((2, 2))
    with pytest.raises(ValueError, match="n_features"):
        solution.transform_robust_scaler(
            X, np.array([0.0]), np.array([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="大于0"):
        solution.transform_robust_scaler(
            X, np.array([0.0, 0.0]), np.array([1.0, 0.0])
        )


@pytest.mark.parametrize(
    "function_name",
    [
        "fit_iqr_bounds",
        "detect_outliers",
        "summarize_outliers",
        "clip_outliers",
        "fit_robust_scaler",
        "transform_robust_scaler",
    ],
)
def test_student_entry_points_remain_unimplemented(function_name: str) -> None:
    function = getattr(starter, function_name)
    X = np.ones((2, 2))
    if function_name in {"fit_iqr_bounds", "fit_robust_scaler"}:
        args = (X,)
    elif function_name == "summarize_outliers":
        args = (np.zeros((2, 2), dtype=bool),)
    elif function_name == "transform_robust_scaler":
        args = (X, np.zeros(2), np.ones(2))
    else:
        args = (X, np.zeros(2), np.ones(2))
    with pytest.raises(NotImplementedError):
        function(*args)


def test_guided_demo_runs_end_to_end() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "train/test shapes: (5, 3) (2, 3)" in result.stdout
    assert "train outlier mask first column: [False False False False  True]" in result.stdout
    assert "test feature outlier counts: [1 1 1]" in result.stdout
    assert "safe IQR scales: [2. 2. 1.]" in result.stdout
