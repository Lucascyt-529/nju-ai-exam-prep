import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "02_machine_learning" / "00_model_evaluation" / "reference" / "solution.py"
spec = importlib.util.spec_from_file_location("cost_curve_solution", SOLUTION)
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def example():
    return np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])


def test_positive_probability_cost_matches_hand_formula():
    probabilities = np.array([0.0, 0.25, 0.5, 1.0])
    expected = probabilities * 4 / (probabilities * 4 + (1 - probabilities) * 2)
    np.testing.assert_allclose(solution.positive_probability_cost(probabilities, 2, 4), expected)


def test_equal_costs_leave_positive_probability_unchanged():
    probabilities = np.linspace(0, 1, 6)
    np.testing.assert_allclose(solution.positive_probability_cost(probabilities, 3, 3), probabilities)


def test_cost_line_has_fpr_and_fnr_as_endpoints():
    x = np.array([0.0, 0.25, 1.0])
    lines = solution.cost_curve_lines(np.array([0.2]), np.array([0.7]), x)
    np.testing.assert_allclose(lines[0], [0.2, 0.225, 0.3])


def test_cost_lines_shape_and_range():
    lines = solution.cost_curve_lines(np.array([0, .5, 1]), np.array([0, .5, 1]), np.linspace(0, 1, 11))
    assert lines.shape == (3, 11)
    assert np.all((lines >= 0) & (lines <= 1))


def test_lower_envelope_selects_best_threshold_at_each_cost():
    y, scores = example(); x = np.linspace(0, 1, 21)
    result = solution.cost_curve_from_scores(y, scores, x)
    np.testing.assert_allclose(result["lower_envelope"], np.min(result["normalized_cost_lines"], axis=0))
    assert result["lower_envelope"][0] == pytest.approx(0.0)
    assert result["lower_envelope"][-1] == pytest.approx(0.0)


def test_cost_curve_reuses_roc_points_and_has_nonnegative_area():
    y, scores = example(); x = np.linspace(0, 1, 101)
    result = solution.cost_curve_from_scores(y, scores, x)
    fpr, tpr, thresholds = solution.roc_curve_points(y, scores)
    np.testing.assert_allclose(result["fpr"], fpr); np.testing.assert_allclose(result["tpr"], tpr)
    np.testing.assert_allclose(result["thresholds"], thresholds)
    assert 0 <= result["area"] <= 1


def test_one_cost_point_has_zero_numeric_area_but_valid_envelope():
    y, scores = example(); result = solution.cost_curve_from_scores(y, scores, np.array([.5]))
    assert result["area"] == 0.0 and result["lower_envelope"].shape == (1,)


@pytest.mark.parametrize("probabilities", [np.array([]), np.array([[.5]]), np.array([-.1]), np.array([np.nan])])
def test_bad_probability_cost_inputs_are_rejected(probabilities):
    with pytest.raises(ValueError): solution.positive_probability_cost(probabilities, 1, 1)


def test_bad_error_costs_are_rejected():
    probabilities = np.array([.5])
    for false_positive, false_negative in ((-1, 1), (1, -1), (0, 0), (np.inf, 1)):
        with pytest.raises(ValueError): solution.positive_probability_cost(probabilities, false_positive, false_negative)


@pytest.mark.parametrize("x", [np.array([.5, .4]), np.array([-1.0]), np.array([[.5]]), np.array([np.nan])])
def test_bad_cost_axis_is_rejected(x):
    with pytest.raises(ValueError): solution.cost_curve_lines(np.array([.2]), np.array([.7]), x)


def test_bad_rates_are_rejected():
    x = np.array([0.0, 1.0])
    with pytest.raises(ValueError): solution.cost_curve_lines(np.array([.2, .3]), np.array([.7]), x)
    with pytest.raises(ValueError): solution.cost_curve_lines(np.array([1.2]), np.array([.7]), x)
    with pytest.raises(ValueError): solution.cost_curve_lines(np.array([.2]), np.array([-.1]), x)
