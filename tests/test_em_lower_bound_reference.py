import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "07_bayesian_classifiers" / "06_em"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("em_bound_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def data():
    return np.array([1, 2, 1, 8, 9, 8]), np.full(6, 10)


def parameters():
    return np.array([0.4, 0.6]), np.array([0.25, 0.75])


def test_expected_complete_log_likelihood_matches_direct_sum() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    responsibilities = np.full((6, 2), 0.5)
    logs = solution._component_logs(
        heads, tosses, mixing, probabilities, include_combination=True
    )
    expected = float(np.sum(responsibilities * logs))
    assert solution.expected_complete_log_likelihood(
        heads, tosses, responsibilities, mixing, probabilities
    ) == pytest.approx(expected)


def test_entropy_uses_zero_log_zero_limit() -> None:
    responsibilities = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert solution.responsibility_entropy(responsibilities) == pytest.approx(
        np.log(2.0)
    )


def test_e_step_posterior_makes_bound_tight() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    responsibilities = solution.expectation_step(
        heads, tosses, mixing, probabilities
    )
    likelihood = solution.observed_log_likelihood(
        heads, tosses, mixing, probabilities
    )
    bound = solution.evidence_lower_bound(
        heads, tosses, responsibilities, mixing, probabilities
    )
    assert bound == pytest.approx(likelihood, abs=1e-12)
    assert solution.posterior_kl_gap(
        heads, tosses, responsibilities, mixing, probabilities
    ) == pytest.approx(0.0, abs=1e-12)


def test_arbitrary_responsibilities_give_strict_lower_bound() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    responsibilities = np.full((6, 2), 0.5)
    likelihood = solution.observed_log_likelihood(
        heads, tosses, mixing, probabilities
    )
    bound = solution.evidence_lower_bound(
        heads, tosses, responsibilities, mixing, probabilities
    )
    assert bound < likelihood


def test_likelihood_minus_bound_equals_posterior_kl_gap() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    responsibilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9], [0.3, 0.7]]
    )
    likelihood = solution.observed_log_likelihood(
        heads, tosses, mixing, probabilities
    )
    bound = solution.evidence_lower_bound(
        heads, tosses, responsibilities, mixing, probabilities
    )
    gap = solution.posterior_kl_gap(
        heads, tosses, responsibilities, mixing, probabilities
    )
    assert gap >= -1e-12
    assert likelihood - bound == pytest.approx(gap, abs=1e-12)


def test_m_step_increases_Q_for_fixed_responsibilities() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    responsibilities = solution.expectation_step(
        heads, tosses, mixing, probabilities
    )
    old_q = solution.expected_complete_log_likelihood(
        heads, tosses, responsibilities, mixing, probabilities
    )
    new_mixing, new_probabilities = solution.maximization_step(
        heads, tosses, responsibilities
    )
    new_q = solution.expected_complete_log_likelihood(
        heads, tosses, responsibilities, new_mixing, new_probabilities
    )
    assert new_q >= old_q - 1e-12


def test_one_em_step_satisfies_bound_likelihood_chain() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    report = solution.em_step_diagnostics(
        heads, tosses, mixing, probabilities
    )
    assert report["old_likelihood"] == pytest.approx(report["old_bound"], abs=1e-12)
    assert report["old_bound"] <= report["after_m_bound"] + 1e-12
    assert report["after_m_bound"] <= report["new_likelihood"] + 1e-12
    assert report["new_likelihood"] == pytest.approx(
        report["tight_new_bound"], abs=1e-12
    )


def test_label_swap_leaves_bound_and_likelihood_unchanged() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    responsibilities = solution.expectation_step(
        heads, tosses, mixing, probabilities
    )
    original = solution.evidence_lower_bound(
        heads, tosses, responsibilities, mixing, probabilities
    )
    swapped = solution.evidence_lower_bound(
        heads,
        tosses,
        responsibilities[:, ::-1],
        mixing[::-1],
        probabilities[::-1],
    )
    assert swapped == pytest.approx(original)


def test_bound_functions_reject_bad_responsibilities() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    for responsibilities in [
        np.ones((6, 2)),
        np.ones((6, 1)),
        np.array([[np.nan, 0.0]] * 6),
        np.array([[1.1, -0.1]] * 6),
    ]:
        with pytest.raises(ValueError):
            solution.evidence_lower_bound(
                heads, tosses, responsibilities, mixing, probabilities
            )


def test_diagnostics_do_not_modify_parameters() -> None:
    heads, tosses = data()
    mixing, probabilities = parameters()
    originals = (heads.copy(), tosses.copy(), mixing.copy(), probabilities.copy())
    solution.em_step_diagnostics(heads, tosses, mixing, probabilities)
    for actual, expected in zip(
        (heads, tosses, mixing, probabilities), originals, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_guided_demo_reports_three_bound_checks() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "old likelihood equals tight bound: True" in result.stdout
    assert "M step raises fixed-responsibility bound: True" in result.stdout
    assert "new likelihood equals new tight bound: True" in result.stdout
