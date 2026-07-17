import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "07_bayesian_classifiers" / "05_bayesian_network"
spec = importlib.util.spec_from_file_location("bn_learning_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def sprinkler_network():
    parents = {"Cloudy": (), "Sprinkler": ("Cloudy",), "Rain": ("Cloudy",), "WetGrass": ("Sprinkler", "Rain")}
    cpt = {"Cloudy": np.array(0.5), "Sprinkler": np.array([0.5, 0.1]), "Rain": np.array([0.2, 0.8]), "WetGrass": np.array([[0.0, 0.9], [0.9, 0.99]])}
    return solution.build_binary_network(parents, cpt)


def dependent_data(repeats=100):
    return np.tile(np.array([[0, 0], [1, 1]], dtype=int), (repeats, 1))


def test_fit_binary_cpts_matches_smoothed_hand_counts():
    data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=int)
    model = solution.fit_binary_cpts(data, ("A", "B"), {"A": (), "B": ("A",)}, alpha=1.0)
    assert model["probability_one"]["A"] == pytest.approx(0.5)
    np.testing.assert_allclose(model["probability_one"]["B"], [0.25, 0.75])


def test_unseen_parent_configuration_uses_neutral_half_without_smoothing():
    data = np.array([[0, 0, 0], [0, 1, 1]], dtype=int)
    model = solution.fit_binary_cpts(data, ("A", "B", "C"), {"A": (), "B": (), "C": ("A", "B")})
    assert model["probability_one"]["C"][1, 0] == pytest.approx(0.5)
    assert model["probability_one"]["C"][1, 1] == pytest.approx(0.5)


def test_parameter_count_counts_one_free_binary_parameter_per_parent_configuration():
    assert solution.parameter_count(sprinkler_network()) == 1 + 2 + 2 + 4


def test_data_log_likelihood_matches_hand_probability_product():
    model = solution.build_binary_network({"A": (), "B": ("A",)}, {"A": np.array(0.5), "B": np.array([0.2, 0.8])})
    data = np.array([[0, 0], [1, 1]], dtype=int)
    assert solution.data_log_likelihood(model, data, ("A", "B")) == pytest.approx(2 * np.log(0.4))


def test_bic_prefers_strong_dependency_despite_extra_parameter():
    data = dependent_data()
    empty = solution.fit_binary_cpts(data, ("A", "B"), {"A": (), "B": ()})
    edge = solution.fit_binary_cpts(data, ("A", "B"), {"A": (), "B": ("A",)})
    assert solution.bic_score(edge, data, ("A", "B")) < solution.bic_score(empty, data, ("A", "B"))


def test_greedy_ordered_structure_adds_forward_dependency_and_decreases_score():
    result = solution.greedy_ordered_structure(dependent_data(), ("A", "B"), order=("A", "B"), max_parents=1)
    assert result["parents"] == {"A": (), "B": ("A",)}
    assert result["score_history"].shape == (2,)
    assert np.all(np.diff(result["score_history"]) < 0)


def test_zero_max_parents_keeps_empty_graph():
    result = solution.greedy_ordered_structure(dependent_data(), ("A", "B"), max_parents=0)
    assert result["parents"] == {"A": (), "B": ()}
    assert result["score_history"].shape == (1,)


def test_markov_blanket_probability_matches_full_joint_ratio():
    model = sprinkler_network()
    state = {"Cloudy": 1, "Sprinkler": 0, "Rain": 0, "WetGrass": 1}
    probability = solution.markov_blanket_probability_one(model, "Rain", state)
    masses = []
    for value in (0, 1):
        candidate = {**state, "Rain": value}
        masses.append(solution.joint_probability(model, candidate))
    assert probability == pytest.approx(masses[1] / sum(masses))


def test_gibbs_query_is_close_to_exact_posterior_on_small_network():
    model = sprinkler_network()
    exact = solution.query_posterior(model, "Rain", {"WetGrass": 1})
    approximate = solution.gibbs_query(model, "Rain", {"WetGrass": 1}, n_samples=8000, burn_in=1000, random_state=11)
    np.testing.assert_allclose(approximate["posterior"], exact, atol=0.035)
    assert approximate["query_samples"].shape == (8000,)
    assert approximate["final_state"]["WetGrass"] == 1


def test_gibbs_query_is_repeatable_and_does_not_mutate_evidence():
    model = sprinkler_network(); evidence = {"WetGrass": 1}; original = dict(evidence)
    first = solution.gibbs_query(model, "Rain", evidence, n_samples=100, burn_in=20, random_state=4)
    second = solution.gibbs_query(model, "Rain", evidence, n_samples=100, burn_in=20, random_state=4)
    np.testing.assert_array_equal(first["query_samples"], second["query_samples"])
    assert evidence == original


@pytest.mark.parametrize("bad", [np.array([0, 1]), np.array([[0, 2]]), np.empty((0, 2))])
def test_learning_rejects_bad_data(bad):
    with pytest.raises(ValueError):
        solution.fit_binary_cpts(bad, ("A", "B"), {"A": (), "B": ()})


def test_learning_rejects_bad_names_structure_alpha_and_order():
    data = dependent_data(2)
    with pytest.raises(ValueError): solution.fit_binary_cpts(data, ("A", "A"), {"A": (), "B": ()})
    with pytest.raises(ValueError): solution.fit_binary_cpts(data, ("A", "B"), {"A": (), "B": ()}, alpha=-1)
    with pytest.raises(ValueError): solution.greedy_ordered_structure(data, ("A", "B"), order=("A", "A"))
    with pytest.raises(ValueError): solution.greedy_ordered_structure(data, ("A", "B"), max_parents=-1)


def test_gibbs_rejects_bad_query_evidence_and_sampling_options():
    model = sprinkler_network()
    with pytest.raises(ValueError): solution.gibbs_query(model, "Rain", {"Rain": 1})
    with pytest.raises(ValueError): solution.gibbs_query(model, "Unknown", {})
    with pytest.raises(ValueError): solution.gibbs_query(model, "Rain", {}, n_samples=0)
    with pytest.raises(ValueError): solution.gibbs_query(model, "Rain", {}, burn_in=-1)
    with pytest.raises(ValueError): solution.gibbs_query(model, "Rain", {}, random_state=1.5)


def test_guided_demo_reports_learning_and_approximate_inference():
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "Gibbs P(Rain|WetGrass=1):" in result.stdout
    assert "learned parents: {'A': (), 'B': ('A',)}" in result.stdout
    assert "BIC decreases: True" in result.stdout
