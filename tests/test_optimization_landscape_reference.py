import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "05_optimization_landscape"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("optimization_landscape_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_objective_gradient_and_curvature_match_formulas() -> None:
    assert solution.objective(2.0) == pytest.approx(9.4)
    assert solution.gradient(2.0) == pytest.approx(24.2)
    assert solution.curvature(2.0) == pytest.approx(44.0)


def test_three_stationary_points_are_real_sorted_and_stationary() -> None:
    points = solution.stationary_points()
    assert len(points) == 3
    assert points == sorted(points)
    assert all(abs(solution.gradient(point)) < 1e-10 for point in points)


def test_stationary_points_have_two_minima_and_one_maximum() -> None:
    classifications = [
        solution.classify_stationary_point(point) for point in solution.stationary_points()
    ]
    assert classifications == ["local_minimum", "local_maximum", "local_minimum"]


def test_left_minimum_is_deeper_and_global_for_this_polynomial() -> None:
    left, _, right = solution.stationary_points()
    point, value = solution.global_minimum_stationary_point()
    assert solution.objective(left) < solution.objective(right)
    assert point == pytest.approx(left)
    assert value == pytest.approx(solution.objective(left))


@pytest.mark.parametrize(
    "start, expected_index",
    [(-2.0, 0), (-0.2, 0), (0.2, 2), (2.0, 2)],
)
def test_initialization_reaches_expected_basin(start: float, expected_index: int) -> None:
    positions, values = solution.gradient_descent(start)
    expected = solution.stationary_points()[expected_index]
    assert positions[-1] == pytest.approx(expected, abs=1e-8)
    assert values[-1] == pytest.approx(solution.objective(expected), abs=1e-10)


def test_chosen_learning_rate_has_nonincreasing_objective() -> None:
    for start in [-2.0, -0.2, 0.2, 2.0]:
        _, values = solution.gradient_descent(start)
        assert np.all(np.diff(values) <= 1e-12)


def test_zero_steps_returns_only_initial_state() -> None:
    positions, values = solution.gradient_descent(0.2, max_steps=0)
    assert positions == [0.2]
    assert values == [solution.objective(0.2)]


def test_summary_is_deterministic_and_does_not_modify_starts() -> None:
    starts = [2.0, -2.0]
    original = starts.copy()
    first = solution.summarize_starts(starts)
    second = solution.summarize_starts(starts)
    assert first == second
    assert starts == original
    assert first[0]["value"] > first[1]["value"]


def test_nonstationary_point_is_not_misclassified() -> None:
    with pytest.raises(ValueError):
        solution.classify_stationary_point(0.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.objective(np.nan),
        lambda: solution.objective(True),
        lambda: solution.classify_stationary_point(0.0, tolerance=0.0),
        lambda: solution.gradient_descent(0.0, learning_rate=0.0),
        lambda: solution.gradient_descent(0.0, gradient_tolerance=-1.0),
        lambda: solution.gradient_descent(0.0, max_steps=-1),
        lambda: solution.gradient_descent(0.0, max_steps=True),
        lambda: solution.summarize_starts([]),
        lambda: solution.summarize_starts((0.0, 1.0)),
    ],
)
def test_invalid_optimization_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_guided_demo_shows_both_attraction_basins() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.count("local_minimum") == 2
    assert result.stdout.count("local_maximum") == 1
    assert "-2.0 -> -1.02412 -0.20244" in result.stdout
    assert "2.0 -> 0.973994 0.197434" in result.stdout
