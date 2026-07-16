import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "03_linear_models" / "03_lda" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("lda_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def sample_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_class_means_and_scatter_are_hand_checkable() -> None:
    X, y = sample_data()
    mean0, mean1 = solution.class_means(X, y)
    scatter = solution.within_class_scatter(X, y)
    np.testing.assert_allclose(mean0, [-1.5, 0.0])
    np.testing.assert_allclose(mean1, [1.5, 1.0])
    np.testing.assert_allclose(scatter, [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(scatter, scatter.T)


def test_lda_fits_and_classifies_separable_data() -> None:
    X, y = sample_data()
    weights, threshold = solution.fit_binary_lda(X, y)
    predictions = solution.predict_lda(X, weights, threshold)
    np.testing.assert_array_equal(predictions, y)
    assert solution.project(X, weights).shape == (4,)
    assert solution.fisher_ratio(X, y, weights) > 0


def test_singular_scatter_and_duplicate_feature_are_supported() -> None:
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    X = np.column_stack((x, x))
    y = np.array([0, 0, 1, 1])
    weights, threshold = solution.fit_binary_lda(X, y)
    np.testing.assert_array_equal(solution.predict_lda(X, weights, threshold), y)
    assert np.all(np.isfinite(weights))


def test_translation_changes_threshold_not_predictions() -> None:
    X, y = sample_data()
    weights, threshold = solution.fit_binary_lda(X, y, regularization=0.1)
    offset = np.array([100.0, -50.0])
    shifted_weights, shifted_threshold = solution.fit_binary_lda(
        X + offset, y, regularization=0.1
    )
    np.testing.assert_allclose(shifted_weights, weights)
    np.testing.assert_array_equal(
        solution.predict_lda(X, weights, threshold),
        solution.predict_lda(X + offset, shifted_weights, shifted_threshold),
    )


def test_scaling_direction_and_threshold_preserves_predictions_and_ratio() -> None:
    X, y = sample_data()
    weights, threshold = solution.fit_binary_lda(X, y, regularization=0.1)
    original = solution.predict_lda(X, weights, threshold)
    scaled = solution.predict_lda(X, weights * 10, threshold * 10)
    np.testing.assert_array_equal(original, scaled)
    assert solution.fisher_ratio(X, y, weights * 10) == pytest.approx(
        solution.fisher_ratio(X, y, weights)
    )


def test_zero_within_class_variance_has_infinite_fisher_ratio() -> None:
    X = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    weights, _ = solution.fit_binary_lda(X, y, regularization=0.1)
    assert np.isinf(solution.fisher_ratio(X, y, weights))


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.fit_binary_lda(np.ones((3, 1)), np.array([0, 0, 0])),
        lambda: solution.fit_binary_lda(np.ones((3, 1)), np.array([0, 1, 2])),
        lambda: solution.fit_binary_lda(np.array([[0.0], [0.0]]), np.array([0, 1])),
        lambda: solution.fit_binary_lda(*sample_data(), regularization=-1.0),
        lambda: solution.predict_lda(np.ones((2, 2)), np.ones(3), 0.0),
    ],
)
def test_invalid_lda_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()
