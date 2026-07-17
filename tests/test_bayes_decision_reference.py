import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "07_bayesian_classifiers" / "01_bayes_decision"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("bayes_decision_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def test_posterior_from_priors_and_likelihoods_matches_hand_calculation() -> None:
    priors = np.array([0.8, 0.2])
    likelihoods = np.array([[0.2, 0.3], [0.1, 0.9]])
    posterior = solution.posterior_from_likelihoods(priors, likelihoods)
    np.testing.assert_allclose(posterior[0], [0.16 / 0.22, 0.06 / 0.22])
    np.testing.assert_allclose(posterior[1], [0.08 / 0.26, 0.18 / 0.26])
    np.testing.assert_allclose(np.sum(posterior, axis=1), 1.0)


def test_likelihood_scaling_per_sample_does_not_change_posterior() -> None:
    priors = np.array([0.4, 0.6])
    likelihoods = np.array([[0.2, 0.5], [0.8, 0.1]])
    scaled = likelihoods * np.array([[10.0], [0.25]])
    np.testing.assert_allclose(
        solution.posterior_from_likelihoods(priors, likelihoods),
        solution.posterior_from_likelihoods(priors, scaled),
    )


def test_zero_one_loss_has_zero_diagonal_and_one_elsewhere() -> None:
    loss = solution.zero_one_loss_matrix(3)
    np.testing.assert_array_equal(loss, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def test_conditional_risks_follow_action_rows_true_class_columns() -> None:
    posterior = np.array([[0.7, 0.3]])
    loss = np.array([[0.0, 5.0], [1.0, 0.0]])
    np.testing.assert_allclose(solution.conditional_risks(posterior, loss), [[1.5, 0.7]])


def test_zero_one_minimum_risk_equals_maximum_posterior() -> None:
    posterior = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    loss = solution.zero_one_loss_matrix(3)
    np.testing.assert_array_equal(
        solution.minimum_risk_decisions(posterior, loss),
        solution.maximum_posterior_decisions(posterior),
    )


def test_asymmetric_cost_can_reverse_maximum_posterior_decision() -> None:
    posterior = np.array([[0.7, 0.3]])
    loss = np.array([[0.0, 5.0], [1.0, 0.0]])
    assert solution.maximum_posterior_decisions(posterior)[0] == 0
    assert solution.minimum_risk_decisions(posterior, loss)[0] == 1


def test_ties_choose_earliest_action_deterministically() -> None:
    posterior = np.array([[0.5, 0.5]])
    loss = solution.zero_one_loss_matrix(2)
    assert solution.minimum_risk_decisions(posterior, loss)[0] == 0
    assert solution.maximum_posterior_decisions(posterior)[0] == 0


def test_selected_risks_extract_each_samples_chosen_action() -> None:
    posterior = np.array([[0.7, 0.3], [0.2, 0.8]])
    loss = np.array([[0.0, 5.0], [1.0, 0.0]])
    decisions = np.array([1, 0])
    np.testing.assert_allclose(solution.selected_risks(posterior, loss, decisions), [0.7, 4.0])


def test_minimum_risk_is_no_worse_than_maximum_posterior_under_same_cost() -> None:
    posterior = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]])
    loss = np.array([[0.0, 5.0], [1.0, 0.0]])
    best = solution.minimum_risk_decisions(posterior, loss)
    maximum = solution.maximum_posterior_decisions(posterior)
    assert np.all(
        solution.selected_risks(posterior, loss, best)
        <= solution.selected_risks(posterior, loss, maximum)
    )


def test_binary_threshold_formula_and_decisions_agree() -> None:
    threshold = solution.binary_positive_threshold(1.0, 5.0)
    assert threshold == pytest.approx(1.0 / 6.0)
    positive_probabilities = np.array([0.1, threshold, 0.3])
    posterior = np.column_stack((1.0 - positive_probabilities, positive_probabilities))
    loss = np.array([[0.0, 5.0], [1.0, 0.0]])
    np.testing.assert_array_equal(solution.minimum_risk_decisions(posterior, loss), [0, 0, 1])


def test_functions_do_not_modify_inputs() -> None:
    priors = np.array([0.4, 0.6])
    likelihoods = np.array([[0.2, 0.5]])
    posterior = np.array([[0.7, 0.3]])
    loss = np.array([[0.0, 5.0], [1.0, 0.0]])
    originals = [value.copy() for value in (priors, likelihoods, posterior, loss)]
    solution.posterior_from_likelihoods(priors, likelihoods)
    solution.conditional_risks(posterior, loss)
    for actual, expected in zip((priors, likelihoods, posterior, loss), originals):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "priors",
    [np.array([0.2, 0.2]), np.array([-0.1, 1.1]), np.array([[0.5, 0.5]]), np.array([1.0])],
)
def test_posterior_rejects_invalid_priors(priors) -> None:
    with pytest.raises(ValueError):
        solution.posterior_from_likelihoods(priors, np.ones((1, priors.size)))


def test_posterior_rejects_bad_likelihoods_and_zero_evidence() -> None:
    priors = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        solution.posterior_from_likelihoods(priors, np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError):
        solution.posterior_from_likelihoods(priors, np.array([[1.0, -1.0]]))
    with pytest.raises(ValueError):
        solution.posterior_from_likelihoods(priors, np.ones((2, 3)))


@pytest.mark.parametrize("n_classes", [1, 0, -1, 2.5, True])
def test_zero_one_loss_rejects_invalid_class_count(n_classes) -> None:
    with pytest.raises(ValueError):
        solution.zero_one_loss_matrix(n_classes)


def test_risks_reject_nonprobability_rows_and_bad_loss_matrix() -> None:
    with pytest.raises(ValueError):
        solution.conditional_risks(np.array([[0.2, 0.2]]), np.eye(2))
    with pytest.raises(ValueError):
        solution.conditional_risks(np.array([[0.5, 0.5]]), np.ones((2, 3)))
    with pytest.raises(ValueError):
        solution.conditional_risks(np.array([[0.5, 0.5]]), np.array([[0.0, -1.0], [1.0, 0.0]]))


def test_selected_risks_rejects_invalid_decisions() -> None:
    posterior = np.array([[0.5, 0.5]])
    loss = np.eye(2)
    with pytest.raises(ValueError):
        solution.selected_risks(posterior, loss, np.array([2]))
    with pytest.raises(ValueError):
        solution.selected_risks(posterior, loss, np.array([0.0]))


@pytest.mark.parametrize("costs", [(0.0, 1.0), (1.0, -1.0), (np.inf, 1.0), (True, 1.0)])
def test_binary_threshold_rejects_invalid_costs(costs) -> None:
    with pytest.raises(ValueError):
        solution.binary_positive_threshold(*costs)


def test_guided_demo_runs_and_shows_decision_difference() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "posterior shape: (2, 2)" in result.stdout
    assert "maximum posterior: [0, 1]" in result.stdout
    assert "minimum risk: [1, 1]" in result.stdout
    assert "positive threshold: 0.1667" in result.stdout
