import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "02_model_evaluation_selection" / "05_comparison_tests" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("comparison_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_binomial_tail_known_values() -> None:
    assert solution.binomial_lower_tail(0, 3, 0.5) == pytest.approx(0.125)
    assert solution.binomial_lower_tail(1, 3, 0.5) == pytest.approx(0.5)
    assert solution.binomial_lower_tail(3, 3, 0.5) == pytest.approx(1.0)


def test_paired_and_corrected_t_statistics() -> None:
    a = np.array([0.82, 0.78, 0.85, 0.80, 0.84])
    b = np.array([0.79, 0.77, 0.80, 0.81, 0.79])
    differences = a - b
    expected = differences.mean() / (differences.std(ddof=1) / np.sqrt(5))
    assert solution.paired_t_statistic(a, b) == pytest.approx(expected)
    corrected = differences.mean() / np.sqrt(
        (1 / 5 + 20 / 80) * differences.var(ddof=1)
    )
    assert solution.corrected_resampled_t_statistic(differences, 80, 20) == pytest.approx(corrected)


def test_zero_variance_t_edge_cases() -> None:
    assert solution.paired_t_statistic(np.ones(3), np.ones(3)) == 0.0
    assert np.isposinf(solution.paired_t_statistic(np.ones(3), np.zeros(3)))


def test_mcnemar_counts_and_exact_probability() -> None:
    truth = np.array([1, 1, 1, 0, 0, 0])
    a = np.array([1, 1, 0, 0, 1, 0])
    b = np.array([1, 0, 1, 0, 0, 1])
    assert solution.mcnemar_disagreements(truth, a, b) == (2, 2)
    assert solution.mcnemar_exact_p_value(1, 9) == pytest.approx(22 / 1024)
    assert solution.mcnemar_exact_p_value(0, 0) == 1.0


def test_ranks_handle_direction_and_ties() -> None:
    scores = np.array([[0.9, 0.8, 0.8], [0.7, 0.9, 0.8]])
    ranks = solution.ranks_per_dataset(scores, higher_is_better=True)
    np.testing.assert_allclose(ranks, [[1.0, 2.5, 2.5], [3.0, 1.0, 2.0]])
    np.testing.assert_allclose(ranks.sum(axis=1), [6.0, 6.0])


def test_friedman_statistic_known_ranks() -> None:
    ranks = np.array([[1, 2, 3], [1, 3, 2], [1, 2, 3], [2, 1, 3]], dtype=float)
    averages, statistic = solution.friedman_statistic(ranks)
    np.testing.assert_allclose(averages, [1.25, 2.0, 2.75])
    assert statistic == pytest.approx(4.5)


def test_nemenyi_critical_difference_and_pairs() -> None:
    cd = solution.nemenyi_critical_difference(3, 10, 2.343)
    assert cd == pytest.approx(2.343 * np.sqrt(12 / 60))
    pairs = solution.nemenyi_significant_pairs(
        ["A", "B", "C"], np.array([1.0, 2.0, 3.0]), 1.5
    )
    assert pairs == [("A", "C")]


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.binomial_lower_tail(4, 3, 0.5),
        lambda: solution.paired_t_statistic(np.ones(2), np.ones(3)),
        lambda: solution.corrected_resampled_t_statistic(np.ones(2), 0, 1),
        lambda: solution.mcnemar_exact_p_value(-1, 2),
        lambda: solution.ranks_per_dataset(np.ones((1, 3)), higher_is_better=True),
        lambda: solution.friedman_statistic(np.array([[1.0, 1.0], [1.0, 1.0]])),
        lambda: solution.nemenyi_critical_difference(1, 10, 2.0),
    ],
)
def test_invalid_comparison_inputs_are_rejected(call) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()
