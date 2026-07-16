import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = (
    ROOT
    / "watermelon_book"
    / "03_linear_models"
    / "05_class_imbalance"
    / "reference"
    / "solution.py"
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("class_imbalance_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def binary_labels() -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, 1, 1])


def test_class_counts_and_balanced_weights_are_hand_checkable() -> None:
    y = binary_labels()
    classes, counts = solution.class_counts(y)
    weight_classes, class_weights = solution.balanced_class_weights(y)
    sample_weights = solution.balanced_sample_weights(y)
    np.testing.assert_array_equal(classes, [0, 1])
    np.testing.assert_array_equal(counts, [6, 2])
    np.testing.assert_array_equal(weight_classes, classes)
    np.testing.assert_allclose(class_weights, [2 / 3, 2.0])
    assert sample_weights.mean() == pytest.approx(1.0)
    assert sample_weights[y == 0].sum() == pytest.approx(4.0)
    assert sample_weights[y == 1].sum() == pytest.approx(4.0)


def test_random_undersampling_is_balanced_unique_and_reproducible() -> None:
    y = binary_labels()
    first = solution.random_undersample_indices(y, random_state=7)
    second = solution.random_undersample_indices(y, random_state=7)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 4
    _, counts = np.unique(y[first], return_counts=True)
    np.testing.assert_array_equal(counts, [2, 2])


def test_random_oversampling_is_balanced_retains_original_and_reproducible() -> None:
    y = binary_labels()
    first = solution.random_oversample_indices(y, random_state=7)
    second = solution.random_oversample_indices(y, random_state=7)
    np.testing.assert_array_equal(first, second)
    assert set(range(len(y))).issubset(set(first.tolist()))
    _, counts = np.unique(y[first], return_counts=True)
    np.testing.assert_array_equal(counts, [6, 6])


def test_sampling_supports_more_than_two_classes() -> None:
    y = np.array([10, 10, 10, 10, 20, 20, 30])
    under = solution.random_undersample_indices(y, random_state=3)
    over = solution.random_oversample_indices(y, random_state=3)
    np.testing.assert_array_equal(np.unique(y[under], return_counts=True)[1], [1, 1, 1])
    np.testing.assert_array_equal(np.unique(y[over], return_counts=True)[1], [4, 4, 4])


def test_same_sampling_indices_keep_features_and_labels_aligned() -> None:
    y = binary_labels()
    X = np.column_stack((np.arange(len(y)), y))
    indices = solution.random_oversample_indices(y, random_state=9)
    np.testing.assert_array_equal(X[indices, 1], y[indices])


def test_higher_false_negative_cost_lowers_threshold_and_predicts_more_positives() -> None:
    probabilities = np.array([0.1, 0.3, 0.6])
    ordinary = solution.cost_sensitive_threshold(1.0, 1.0)
    recall_focused = solution.cost_sensitive_threshold(1.0, 4.0)
    assert ordinary == pytest.approx(0.5)
    assert recall_focused == pytest.approx(0.2)
    assert solution.predict_with_threshold(probabilities, recall_focused).sum() > solution.predict_with_threshold(
        probabilities, ordinary
    ).sum()


def test_prior_correction_identity_boundaries_and_direction() -> None:
    probabilities = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    unchanged = solution.correct_prior_shift(
        probabilities, source_positive_prior=0.4, target_positive_prior=0.4
    )
    rarer_positive = solution.correct_prior_shift(
        probabilities, source_positive_prior=0.5, target_positive_prior=0.1
    )
    np.testing.assert_allclose(unchanged, probabilities)
    assert rarer_positive[0] == 0.0
    assert rarer_positive[-1] == 1.0
    assert np.all(rarer_positive[1:-1] < probabilities[1:-1])


def test_prior_correction_and_original_probability_threshold_are_equivalent() -> None:
    probabilities = np.array([0.1, 0.79, 0.81, 0.95])
    corrected = solution.correct_prior_shift(
        probabilities, source_positive_prior=0.5, target_positive_prior=0.2
    )
    original_threshold = solution.prior_shift_decision_threshold(
        source_positive_prior=0.5, target_positive_prior=0.2
    )
    assert original_threshold == pytest.approx(0.8)
    np.testing.assert_array_equal(
        solution.predict_with_threshold(corrected, 0.5),
        solution.predict_with_threshold(probabilities, original_threshold),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.class_counts(np.array([1, 1, 1])),
        lambda: solution.class_counts(np.array([0.0, np.nan, 1.0])),
        lambda: solution.random_undersample_indices(np.array([[0, 1], [1, 0]])),
        lambda: solution.cost_sensitive_threshold(0.0, 1.0),
        lambda: solution.cost_sensitive_threshold(1.0, np.inf),
        lambda: solution.predict_with_threshold(np.array([0.2, 1.2]), 0.5),
        lambda: solution.predict_with_threshold(np.array([0.2, 0.8]), -0.1),
        lambda: solution.correct_prior_shift(
            np.array([0.5]), source_positive_prior=0.0, target_positive_prior=0.2
        ),
        lambda: solution.prior_shift_decision_threshold(
            source_positive_prior=0.5, target_positive_prior=1.0
        ),
    ],
)
def test_invalid_class_imbalance_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()
