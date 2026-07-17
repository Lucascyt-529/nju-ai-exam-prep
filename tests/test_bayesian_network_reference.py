import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "07_bayesian_classifiers" / "05_bayesian_network"
spec = importlib.util.spec_from_file_location("bayesian_network_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def network():
    parents = {"Cloudy": (), "Sprinkler": ("Cloudy",), "Rain": ("Cloudy",), "WetGrass": ("Sprinkler", "Rain")}
    cpt = {"Cloudy": np.array(0.5), "Sprinkler": np.array([0.5, 0.1]), "Rain": np.array([0.2, 0.8]), "WetGrass": np.array([[0.0, 0.9], [0.9, 0.99]])}
    return solution.build_binary_network(parents, cpt)


def test_topological_order_places_every_parent_before_child() -> None:
    model = network(); order = model["order"]
    assert order == ("Cloudy", "Rain", "Sprinkler", "WetGrass")
    positions = {name: i for i, name in enumerate(order)}
    for child, parents in model["parents"].items():
        assert all(positions[parent] < positions[child] for parent in parents)


def test_build_copies_cpts_and_parent_tuples() -> None:
    parents = {"A": (), "B": ("A",)}; table = np.array([0.2, 0.8])
    model = solution.build_binary_network(parents, {"A": np.array(0.5), "B": table})
    table[0] = 0.9; parents["B"] = ()
    assert model["probability_one"]["B"][0] == pytest.approx(0.2)
    assert model["parents"]["B"] == ("A",)


def test_joint_probability_matches_hand_factorization() -> None:
    assignment = {"Cloudy": 1, "Sprinkler": 0, "Rain": 1, "WetGrass": 1}
    assert solution.joint_probability(network(), assignment) == pytest.approx(0.5 * 0.9 * 0.8 * 0.9)


def test_all_assignments_are_complete_unique_and_normalized() -> None:
    model = network(); assignments = list(solution.all_assignments(model))
    assert len(assignments) == 16
    assert len({tuple(item[name] for name in model["order"]) for item in assignments}) == 16
    assert sum(solution.joint_probability(model, item) for item in assignments) == pytest.approx(1.0)


def test_evidence_probability_matches_known_value() -> None:
    assert solution.evidence_probability(network(), {"WetGrass": 1}) == pytest.approx(0.6471)


def test_query_posterior_matches_joint_ratio_and_normalizes() -> None:
    model = network(); posterior = solution.query_posterior(model, "Rain", {"WetGrass": 1})
    expected_one = solution.evidence_probability(model, {"Rain": 1, "WetGrass": 1}) / solution.evidence_probability(model, {"WetGrass": 1})
    assert posterior.shape == (2,)
    assert posterior[1] == pytest.approx(expected_one)
    assert posterior.sum() == pytest.approx(1.0)


def test_empty_evidence_probability_is_one_and_prior_query_works() -> None:
    model = network()
    assert solution.evidence_probability(model, {}) == pytest.approx(1.0)
    np.testing.assert_allclose(solution.query_posterior(model, "Cloudy", {}), [0.5, 0.5])


def test_parent_axis_order_changes_cpt_lookup_as_documented() -> None:
    model = network()
    # WetGrass父顺序是(Sprinkler,Rain)，索引(1,0)读取0.9。
    assignment = {"Cloudy": 0, "Sprinkler": 1, "Rain": 0, "WetGrass": 1}
    expected = 0.5 * 0.5 * 0.8 * 0.9
    assert solution.joint_probability(model, assignment) == pytest.approx(expected)


def test_cycle_self_parent_unknown_parent_and_duplicate_parent_are_rejected() -> None:
    with pytest.raises(ValueError, match="有向环"):
        solution.topological_order({"A": ("B",), "B": ("A",)})
    with pytest.raises(ValueError): solution.topological_order({"A": ("A",)})
    with pytest.raises(ValueError): solution.topological_order({"A": ("B",)})
    with pytest.raises(ValueError): solution.topological_order({"A": (), "B": ("A", "A")})


def test_cpt_keys_shapes_probabilities_and_numeric_type_are_validated() -> None:
    parents = {"A": (), "B": ("A",)}
    with pytest.raises(ValueError): solution.build_binary_network(parents, {"A": np.array(0.5)})
    with pytest.raises(ValueError): solution.build_binary_network(parents, {"A": np.array([0.5]), "B": np.array([0.2, 0.8])})
    with pytest.raises(ValueError): solution.build_binary_network(parents, {"A": np.array(0.5), "B": np.array([0.2, 1.2])})
    with pytest.raises(ValueError): solution.build_binary_network(parents, {"A": np.array(0.5), "B": np.array(["x", "y"])})


def test_assignment_requires_all_and_only_binary_integer_values() -> None:
    model = network(); good = {name: 0 for name in model["order"]}
    for bad in ({"Cloudy": 0}, {**good, "Extra": 0}, {**good, "Rain": 2}, {**good, "Rain": True}):
        with pytest.raises(ValueError): solution.joint_probability(model, bad)


def test_evidence_and_query_validation() -> None:
    model = network()
    with pytest.raises(ValueError): solution.evidence_probability(model, {"Unknown": 1})
    with pytest.raises(ValueError): solution.query_posterior(model, "Unknown", {})
    with pytest.raises(ValueError): solution.query_posterior(model, "Rain", {"Rain": 1})


def test_zero_probability_evidence_has_undefined_posterior() -> None:
    parents = {"A": (), "B": ()}
    model = solution.build_binary_network(parents, {"A": np.array(0.0), "B": np.array(0.5)})
    with pytest.raises(ValueError, match="概率为0"):
        solution.query_posterior(model, "B", {"A": 1})


def test_guided_demo_runs_and_reports_exact_inference() -> None:
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "assignment count: 16" in result.stdout
    assert "joint sum: 1.0" in result.stdout
    assert "P(WetGrass=1): 0.6471" in result.stdout
    assert "P(Rain|WetGrass=1): [0.292072 0.707928]" in result.stdout

