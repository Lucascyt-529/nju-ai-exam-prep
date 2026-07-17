import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "06_support_vector_machines" / "08_surrogate_losses"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("surrogate_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def margins() -> np.ndarray:
    return np.array([-1.0, 0.0, 0.5, 1.0, 2.0])


def test_zero_one_uses_strict_negative_margin_convention() -> None:
    np.testing.assert_array_equal(
        solution.zero_one_losses(margins()), [1.0, 0.0, 0.0, 0.0, 0.0]
    )


def test_hinge_matches_hand_values() -> None:
    np.testing.assert_allclose(
        solution.surrogate_losses(margins(), kind="hinge"),
        [2.0, 1.0, 0.5, 0.0, 0.0],
    )


def test_exponential_matches_definition() -> None:
    np.testing.assert_allclose(
        solution.surrogate_losses(margins(), kind="exponential"),
        np.exp(-margins()),
    )


def test_logistic_is_stable_base_two_form() -> None:
    expected = np.logaddexp(0.0, -margins()) / np.log(2.0)
    np.testing.assert_allclose(
        solution.surrogate_losses(margins(), kind="logistic"), expected
    )
    assert expected[1] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["hinge", "exponential", "logistic"])
def test_surrogates_upper_bound_strict_negative_zero_one(kind: str) -> None:
    values = np.linspace(-4.0, 4.0, 101)
    assert np.all(
        solution.surrogate_losses(values, kind=kind)
        >= solution.zero_one_losses(values) - 1e-12
    )


def test_hinge_gradient_uses_zero_subgradient_at_kink() -> None:
    np.testing.assert_array_equal(
        solution.margin_gradients(margins(), kind="hinge"),
        [-1.0, -1.0, -1.0, 0.0, 0.0],
    )


@pytest.mark.parametrize("kind", ["exponential", "logistic"])
def test_smooth_gradients_match_centered_difference(kind: str) -> None:
    points = np.array([-2.0, -0.5, 0.5, 2.0])
    epsilon = 1e-6
    numerical = (
        solution.surrogate_losses(points + epsilon, kind=kind)
        - solution.surrogate_losses(points - epsilon, kind=kind)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        solution.margin_gradients(points, kind=kind), numerical, rtol=1e-6, atol=1e-8
    )


def test_hinge_has_exact_sparse_activity_outside_margin() -> None:
    active = solution.active_gradient_mask(margins(), kind="hinge")
    np.testing.assert_array_equal(active, [True, True, True, False, False])


@pytest.mark.parametrize("kind", ["exponential", "logistic"])
def test_smooth_losses_remain_active_at_finite_margin(kind: str) -> None:
    active = solution.active_gradient_mask(margins(), kind=kind, tolerance=1e-15)
    assert np.all(active)


@pytest.mark.parametrize("kind", ["hinge", "exponential", "logistic"])
def test_surrogate_is_convex_by_jensen_gap(kind: str) -> None:
    left = np.array([-3.0, -1.0, 0.5])
    right = np.array([2.0, 1.5, 4.0])
    gap = solution.convexity_gap(left, right, 0.3, kind=kind)
    assert np.all(gap <= 1e-12)


def test_regularized_objective_reports_components() -> None:
    report = solution.regularized_objective(
        np.array([3.0, 4.0]), np.array([0.0, 2.0]), 2.0, kind="hinge"
    )
    assert report == pytest.approx(
        {"regularization": 12.5, "empirical_penalty": 2.0, "objective": 14.5}
    )


def test_logistic_stays_finite_where_exponential_overflows() -> None:
    extreme = np.array([-1000.0])
    assert np.isfinite(solution.surrogate_losses(extreme, kind="logistic")[0])
    with pytest.raises(ValueError, match="指数损失"):
        solution.surrogate_losses(extreme, kind="exponential")


@pytest.mark.parametrize("function", [solution.zero_one_losses, solution.surrogate_losses, solution.margin_gradients])
def test_loss_functions_reject_bad_margin_shape(function) -> None:
    with pytest.raises(ValueError):
        function(np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError):
        function(np.array([1.0, np.nan]))


def test_unknown_kind_and_invalid_tolerance_are_rejected() -> None:
    with pytest.raises(ValueError, match="kind"):
        solution.surrogate_losses(np.array([1.0]), kind="unknown")
    with pytest.raises(ValueError):
        solution.active_gradient_mask(np.array([1.0]), tolerance=0.0)


@pytest.mark.parametrize("C", [0.0, -1.0, np.inf, True])
def test_objective_rejects_invalid_C(C) -> None:
    with pytest.raises(ValueError):
        solution.regularized_objective(
            np.array([1.0]), np.array([1.0]), C, kind="hinge"
        )


def test_convexity_gap_rejects_shape_and_mixing_errors() -> None:
    with pytest.raises(ValueError, match="形状"):
        solution.convexity_gap(
            np.array([1.0]), np.array([1.0, 2.0]), 0.5, kind="hinge"
        )
    with pytest.raises(ValueError, match="mixing"):
        solution.convexity_gap(
            np.array([1.0]), np.array([2.0]), 1.1, kind="hinge"
        )


def test_inputs_are_not_modified() -> None:
    values = margins()
    weights = np.array([1.0, 2.0])
    values_before, weights_before = values.copy(), weights.copy()
    for kind in ("hinge", "exponential", "logistic"):
        solution.surrogate_losses(values, kind=kind)
        solution.margin_gradients(values, kind=kind)
        solution.regularized_objective(weights, values, 1.0, kind=kind)
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(weights, weights_before)


def test_guided_demo_runs_and_shows_activity_difference() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "zero-one: [1.0, 0.0, 0.0, 0.0, 0.0]" in result.stdout
    assert "hinge active gradient: [True, True, True, False, False]" in result.stdout
    assert "smooth losses do not" in result.stdout
