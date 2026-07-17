import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "08_ensemble_learning" / "06_boosting_exponential_view"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("boosting_view_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def test_exponential_risk_matches_hand_calculation() -> None:
    y = np.array([-1, 1])
    scores = np.array([-0.5, 1.0])
    assert solution.exponential_risk(y, scores) == pytest.approx(
        0.5 * (np.exp(-0.5) + np.exp(-1.0))
    )


def test_weighted_exponential_risk_uses_distribution() -> None:
    y = np.array([-1, 1])
    scores = np.array([-0.5, 1.0])
    distribution = np.array([0.25, 0.75])
    assert solution.exponential_risk(y, scores, distribution=distribution) == pytest.approx(
        0.25 * np.exp(-0.5) + 0.75 * np.exp(-1.0)
    )


def test_bayes_optimal_score_is_half_log_odds_and_has_correct_sign() -> None:
    probabilities = np.array([0.2, 0.5, 0.8])
    scores = solution.bayes_optimal_additive_score(probabilities)
    np.testing.assert_allclose(scores, 0.5 * np.log(probabilities / (1 - probabilities)))
    np.testing.assert_array_equal(np.sign(scores), [-1.0, 0.0, 1.0])


def test_round_loss_matches_correct_and_error_partition() -> None:
    alpha = 0.7
    error = 0.25
    assert solution.round_exponential_loss(error, alpha) == pytest.approx(
        np.exp(-alpha) * 0.75 + np.exp(alpha) * 0.25
    )


@pytest.mark.parametrize("error", [0.1, 0.25, 0.4, 0.6, 0.9])
def test_optimal_alpha_matches_formula_and_zero_derivative(error: float) -> None:
    alpha = solution.optimal_alpha(error)
    assert alpha == pytest.approx(0.5 * np.log((1 - error) / error))
    epsilon = 1e-6
    derivative = (
        solution.round_exponential_loss(error, alpha + epsilon)
        - solution.round_exponential_loss(error, alpha - epsilon)
    ) / (2 * epsilon)
    assert derivative == pytest.approx(0.0, abs=1e-8)


def test_alpha_sign_reflects_weak_learner_quality() -> None:
    assert solution.optimal_alpha(0.25) > 0
    assert solution.optimal_alpha(0.5) == pytest.approx(0.0)
    assert solution.optimal_alpha(0.75) < 0


def test_update_distribution_returns_normalizer_and_expected_weights() -> None:
    distribution = np.full(4, 0.25)
    y = np.array([-1, -1, 1, 1])
    prediction = np.array([-1, 1, 1, 1])
    alpha = solution.optimal_alpha(0.25)
    updated, normalizer = solution.update_distribution(distribution, y, prediction, alpha)
    raw = distribution * np.exp(-alpha * y * prediction)
    np.testing.assert_allclose(updated, raw / raw.sum())
    assert normalizer == pytest.approx(raw.sum())
    np.testing.assert_allclose(updated, [1 / 6, 1 / 2, 1 / 6, 1 / 6])


def test_distribution_from_scores_matches_sequential_updates() -> None:
    y = np.array([-1, -1, 1, 1])
    predictions = np.array([[-1, 1, 1, 1], [-1, -1, -1, 1]])
    alphas = np.array([solution.optimal_alpha(0.25), solution.optimal_alpha(1 / 6)])
    trace = solution.trace_additive_rounds(y, predictions, alphas)
    reconstructed = solution.distribution_from_scores(
        np.full(4, 0.25), y, trace["scores"][-1]
    )
    np.testing.assert_allclose(reconstructed, trace["distributions"][-1])


def test_exponential_risk_equals_cumulative_normalizer_product() -> None:
    y = np.array([-1, -1, 1, 1])
    predictions = np.array([[-1, 1, 1, 1], [-1, -1, -1, 1]])
    alphas = np.array([solution.optimal_alpha(0.25), solution.optimal_alpha(1 / 6)])
    trace = solution.trace_additive_rounds(y, predictions, alphas)
    np.testing.assert_allclose(trace["risks"], trace["normalizer_products"], atol=1e-12)
    np.testing.assert_allclose(trace["distributions"].sum(axis=1), 1.0)


def test_training_error_is_bounded_by_exponential_risk() -> None:
    y = np.array([-1, -1, 1, 1])
    for scores in [np.zeros(4), np.array([-2.0, 0.2, 0.3, -0.5]), np.array([-3.0, -2.0, 2.0, 3.0])]:
        error, bound = solution.training_error_and_exponential_bound(y, scores)
        assert error <= bound + 1e-12


def test_weighted_resampling_is_reproducible_and_respects_support() -> None:
    distribution = np.array([0.0, 0.1, 0.9])
    first = solution.weighted_resample_indices(distribution, n_samples=100, seed=42)
    second = solution.weighted_resample_indices(distribution, n_samples=100, seed=42)
    np.testing.assert_array_equal(first, second)
    assert 0 not in first
    assert np.count_nonzero(first == 2) > np.count_nonzero(first == 1)


def test_resampling_different_seeds_need_not_match() -> None:
    distribution = np.array([0.5, 0.5])
    first = solution.weighted_resample_indices(distribution, n_samples=20, seed=1)
    second = solution.weighted_resample_indices(distribution, n_samples=20, seed=2)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("error", [0.0, 1.0, -0.1, np.inf, True])
def test_optimal_alpha_rejects_boundary_or_invalid_error(error) -> None:
    with pytest.raises(ValueError):
        solution.optimal_alpha(error)


def test_probability_and_distribution_validation() -> None:
    with pytest.raises(ValueError):
        solution.bayes_optimal_additive_score(np.array([0.0, 0.5]))
    with pytest.raises(ValueError):
        solution.exponential_risk(
            np.array([-1, 1]), np.array([0.0, 0.0]), distribution=np.array([0.2, 0.2])
        )


def test_trace_rejects_prediction_shape_values_and_alpha_count() -> None:
    y = np.array([-1, 1])
    with pytest.raises(ValueError):
        solution.trace_additive_rounds(y, np.array([-1, 1]), np.array([1.0]))
    with pytest.raises(ValueError):
        solution.trace_additive_rounds(y, np.array([[-1, 0]]), np.array([1.0]))
    with pytest.raises(ValueError):
        solution.trace_additive_rounds(y, np.array([[-1, 1]]), np.array([1.0, 2.0]))


def test_inputs_are_not_modified() -> None:
    y = np.array([-1, -1, 1, 1])
    predictions = np.array([[-1, 1, 1, 1], [-1, -1, -1, 1]])
    alphas = np.array([0.5, 0.7])
    before = (y.copy(), predictions.copy(), alphas.copy())
    solution.trace_additive_rounds(y, predictions, alphas)
    for actual, expected in zip((y, predictions, alphas), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_guided_demo_runs_and_confirms_normalizer_identity() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "normalizers:" in result.stdout
    assert "risks:" in result.stdout
    assert "risk equals product Z: True" in result.stdout
