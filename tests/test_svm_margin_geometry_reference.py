import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "06_support_vector_machines" / "01_margin_geometry"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("svm_margin_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def separable_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    y = np.array([-1, -1, 1, 1])
    weights = np.array([1.0, 0.0])
    return X, y, weights, 0.0


def test_scores_predictions_and_shapes() -> None:
    X, _, weights, bias = separable_data()
    scores = solution.decision_scores(X, weights, bias)
    np.testing.assert_allclose(scores, [-2.0, -1.0, 1.0, 2.0])
    np.testing.assert_array_equal(solution.predict_labels(X, weights, bias), [-1, -1, 1, 1])
    assert scores.shape == (4,)


def test_zero_score_tie_predicts_positive_label() -> None:
    X = np.array([[0.0, 0.0]])
    np.testing.assert_array_equal(
        solution.predict_labels(X, np.array([1.0, 0.0]), 0.0), [1]
    )


def test_functional_and_geometric_margins_match_hand_values() -> None:
    X, y, weights, bias = separable_data()
    np.testing.assert_allclose(
        solution.functional_margins(X, y, weights, bias), [2.0, 1.0, 1.0, 2.0]
    )
    np.testing.assert_allclose(
        solution.geometric_margins(X, y, weights, bias), [2.0, 1.0, 1.0, 2.0]
    )


def test_point_distances_ignore_label_sign() -> None:
    X, _, weights, bias = separable_data()
    np.testing.assert_allclose(
        solution.point_to_hyperplane_distances(X, weights, bias), [2.0, 1.0, 1.0, 2.0]
    )


def test_positive_parameter_scaling_changes_only_functional_margin() -> None:
    X, y, weights, bias = separable_data()
    functional = solution.functional_margins(X, y, weights, bias)
    geometric = solution.geometric_margins(X, y, weights, bias)
    scaled_functional = solution.functional_margins(X, y, 3.0 * weights, 3.0 * bias)
    scaled_geometric = solution.geometric_margins(X, y, 3.0 * weights, 3.0 * bias)
    np.testing.assert_allclose(scaled_functional, 3.0 * functional)
    np.testing.assert_allclose(scaled_geometric, geometric)
    np.testing.assert_array_equal(
        solution.predict_labels(X, 3.0 * weights, 3.0 * bias),
        solution.predict_labels(X, weights, bias),
    )


def test_canonical_rescale_sets_minimum_functional_margin_to_one() -> None:
    X, y, weights, bias = separable_data()
    canonical_weights, canonical_bias = solution.canonical_rescale(
        X, y, 3.0 * weights, 3.0 * bias
    )
    np.testing.assert_allclose(canonical_weights, weights)
    assert canonical_bias == pytest.approx(bias)
    assert np.min(
        solution.functional_margins(X, y, canonical_weights, canonical_bias)
    ) == pytest.approx(1.0)


def test_minimum_margin_indices_find_closest_signed_points() -> None:
    X, y, weights, bias = separable_data()
    np.testing.assert_array_equal(
        solution.minimum_margin_indices(X, y, weights, bias), [1, 2]
    )


def test_hard_margin_objective_is_half_squared_norm() -> None:
    assert solution.hard_margin_primal_objective(np.array([3.0, 4.0])) == pytest.approx(12.5)


def test_canonical_rescale_rejects_misclassified_or_boundary_sample() -> None:
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([-1, 1, 1])
    with pytest.raises(ValueError):
        solution.canonical_rescale(X, y, np.array([1.0]), 0.0)


def test_inputs_are_not_modified() -> None:
    X, y, weights, bias = separable_data()
    original_X, original_y, original_weights = X.copy(), y.copy(), weights.copy()
    solution.canonical_rescale(X, y, weights, bias)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.decision_scores(np.array([1.0, 2.0]), np.ones(2), 0.0),
        lambda: solution.decision_scores(np.ones((2, 2)), np.ones(3), 0.0),
        lambda: solution.decision_scores(np.ones((2, 2)), np.ones(2), np.nan),
        lambda: solution.functional_margins(
            np.ones((2, 2)), np.array([0, 1]), np.ones(2), 0.0
        ),
        lambda: solution.functional_margins(
            np.ones((2, 2)), np.array([[-1], [1]]), np.ones(2), 0.0
        ),
        lambda: solution.geometric_margins(
            np.ones((2, 2)), np.array([-1, 1]), np.zeros(2), 0.0
        ),
        lambda: solution.minimum_margin_indices(
            *separable_data(), tolerance=-1.0
        ),
        lambda: solution.hard_margin_primal_objective(np.ones((2, 1))),
    ],
)
def test_invalid_margin_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_guided_demo_shows_scaling_invariance() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "X / y / w: (4, 2) (4,) (2,)" in result.stdout
    assert "functional: [2. 1. 1. 2.]" in result.stdout
    assert "scaled functional: [6. 3. 3. 6.]" in result.stdout
    assert "geometric: [2. 1. 1. 2.]" in result.stdout
    assert "scaled geometric: [2. 1. 1. 2.]" in result.stdout
    assert "minimum-margin indices: [1 2]" in result.stdout
