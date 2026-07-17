import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "07_bayesian_classifiers" / "02_maximum_likelihood"
spec = importlib.util.spec_from_file_location("bayes_mle_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def test_class_prior_mle_sorts_labels_and_normalizes() -> None:
    classes, priors = solution.class_prior_mle(np.array([3, 1, 3, 2, 3, 1]))
    np.testing.assert_array_equal(classes, [1, 2, 3])
    np.testing.assert_allclose(priors, [2 / 6, 1 / 6, 3 / 6])


def test_bernoulli_mle_is_fraction_of_ones() -> None:
    assert solution.bernoulli_mle(np.array([1, 0, 1, 1])) == pytest.approx(0.75)


def test_categorical_probabilities_without_and_with_smoothing() -> None:
    samples = np.array([0, 0, 1])
    categories = np.array([0, 1, 2])
    np.testing.assert_allclose(solution.categorical_probabilities(samples, categories), [2 / 3, 1 / 3, 0])
    np.testing.assert_allclose(solution.categorical_probabilities(samples, categories, alpha=1), [3 / 6, 2 / 6, 1 / 6])


def test_gaussian_mle_uses_n_denominator() -> None:
    mean, variance = solution.gaussian_mle(np.array([1.0, 2.0, 3.0]))
    assert mean == pytest.approx(2.0)
    assert variance == pytest.approx(2.0 / 3.0)
    assert variance != pytest.approx(np.var([1.0, 2.0, 3.0], ddof=1))


def test_multivariate_mle_matches_centered_product_over_n() -> None:
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    mean, covariance = solution.multivariate_gaussian_mle(X)
    np.testing.assert_allclose(mean, [2.0, 4.0])
    centered = X - mean
    np.testing.assert_allclose(covariance, centered.T @ centered / 3)
    assert np.linalg.matrix_rank(covariance) == 1


def test_bernoulli_mle_maximizes_log_likelihood() -> None:
    samples = np.array([1, 0, 1, 1])
    estimate = solution.bernoulli_mle(samples)
    assert solution.bernoulli_log_likelihood(samples, estimate) > solution.bernoulli_log_likelihood(samples, 0.5)


def test_impossible_bernoulli_probability_has_negative_infinite_log_likelihood() -> None:
    assert solution.bernoulli_log_likelihood(np.array([0, 1]), 0.0) == -np.inf
    assert solution.bernoulli_log_likelihood(np.array([0, 1]), 1.0) == -np.inf


def test_estimates_do_not_modify_inputs() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = X.copy()
    solution.multivariate_gaussian_mle(X)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("samples", [np.array([]), np.array([[1.0]]), np.array([np.nan])])
def test_vector_estimators_reject_invalid_samples(samples) -> None:
    with pytest.raises(ValueError):
        solution.gaussian_mle(samples)


def test_bernoulli_and_categories_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        solution.bernoulli_mle(np.array([0, 2]))
    with pytest.raises(ValueError):
        solution.categorical_probabilities(np.array([0, 3]), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        solution.categorical_probabilities(np.array([0]), np.array([0, 0]))


@pytest.mark.parametrize("alpha", [-1.0, np.inf, True])
def test_categorical_rejects_invalid_smoothing(alpha) -> None:
    with pytest.raises(ValueError):
        solution.categorical_probabilities(np.array([0]), np.array([0, 1]), alpha=alpha)


def test_multivariate_rejects_one_dimensional_and_empty_matrices() -> None:
    with pytest.raises(ValueError):
        solution.multivariate_gaussian_mle(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        solution.multivariate_gaussian_mle(np.empty((0, 2)))


def test_guided_demo_runs_and_shows_ddof_difference() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "MLE variance ddof=0: 0.666667" in result.stdout
    assert "unbiased variance ddof=1: 1.0" in result.stdout
    assert "covariance rank: 1" in result.stdout
