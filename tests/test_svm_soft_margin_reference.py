import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "watermelon_book" / "06_support_vector_machines"
SOFT_SOLUTION = CHAPTER / "03_soft_margin" / "reference" / "solution.py"
SMO_SOLUTION = CHAPTER / "02_linear_smo" / "reference" / "solution.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soft = load_module("svm_soft_margin_solution", SOFT_SOLUTION)
smo = load_module("svm_soft_margin_smo", SMO_SOLUTION)


def test_signed_margins_match_hand_calculation() -> None:
    y = np.array([-1, -1, 1, 1, 1])
    scores = np.array([-1.5, -1.0, 0.5, 0.0, -0.25])
    np.testing.assert_allclose(soft.signed_margins(y, scores), [1.5, 1.0, 0.5, 0.0, -0.25])


def test_hinge_loss_is_positive_part_of_one_minus_margin() -> None:
    y = np.array([-1, -1, 1, 1, 1])
    scores = np.array([-1.5, -1.0, 0.5, 0.0, -0.25])
    np.testing.assert_allclose(soft.hinge_losses(y, scores), [0.0, 0.0, 0.5, 1.0, 1.25])


def test_correct_prediction_inside_margin_still_has_loss() -> None:
    assert soft.hinge_losses(np.array([1]), np.array([0.2]))[0] == pytest.approx(0.8)


def test_margin_regions_cover_all_five_cases() -> None:
    margins = np.array([1.5, 1.0, 0.5, 0.0, -0.25])
    assert soft.margin_region_labels(margins).tolist() == [
        soft.OUTSIDE_MARGIN,
        soft.ON_MARGIN,
        soft.INSIDE_MARGIN,
        soft.DECISION_BOUNDARY,
        soft.MISCLASSIFIED,
    ]


def test_margin_regions_use_tolerance_around_boundaries() -> None:
    margins = np.array([1.0 + 5e-7, 5e-7, -5e-7])
    assert soft.margin_region_labels(margins, tolerance=1e-6).tolist() == [
        soft.ON_MARGIN,
        soft.DECISION_BOUNDARY,
        soft.DECISION_BOUNDARY,
    ]


def test_alpha_statuses_distinguish_nonfree_and_bounded_support() -> None:
    alphas = np.array([0.0, 0.25, 1.0])
    assert soft.alpha_status_labels(alphas, 1.0).tolist() == [
        soft.NON_SUPPORT,
        soft.FREE_SUPPORT,
        soft.BOUNDED_SUPPORT,
    ]


def test_kkt_flags_match_three_alpha_positions() -> None:
    margins = np.array([1.4, 1.0, 0.3, -0.5])
    alphas = np.array([0.0, 0.4, 1.0, 1.0])
    np.testing.assert_array_equal(
        soft.kkt_consistency_flags(margins, alphas, 1.0),
        [True, True, True, True],
    )


def test_kkt_flags_detect_wrong_margin_for_each_alpha_position() -> None:
    margins = np.array([0.8, 0.8, 1.2])
    alphas = np.array([0.0, 0.4, 1.0])
    np.testing.assert_array_equal(
        soft.kkt_consistency_flags(margins, alphas, 1.0),
        [False, False, False],
    )


def test_analysis_reports_objective_components_and_counts() -> None:
    report = soft.analyze_soft_margin_solution(
        np.array([2.0, 0.0]),
        np.array([-1, -1, 1, 1, 1]),
        np.array([-1.5, -1.0, 0.5, 0.0, -0.25]),
        np.array([0.0, 0.2, 1.0, 1.0, 1.0]),
        1.0,
    )
    assert report["regularization"] == pytest.approx(2.0)
    assert report["slack_penalty"] == pytest.approx(2.75)
    assert report["objective"] == pytest.approx(4.75)
    assert report["support_count"] == 4
    assert report["margin_violation_count"] == 3
    assert report["misclassified_count"] == 1
    np.testing.assert_array_equal(report["slacks"], report["hinge_losses"])
    assert report["slacks"] is not report["hinge_losses"]


def test_analysis_does_not_modify_inputs() -> None:
    weights = np.array([1.0])
    y = np.array([-1, 1])
    scores = np.array([-1.0, 1.0])
    alphas = np.array([0.5, 0.5])
    originals = [array.copy() for array in (weights, y, scores, alphas)]
    soft.analyze_soft_margin_solution(weights, y, scores, alphas, 1.0)
    for actual, expected in zip((weights, y, scores, alphas), originals):
        np.testing.assert_array_equal(actual, expected)


def c_tradeoff_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[-3.0], [-2.0], [-1.0], [0.2], [1.0], [2.0]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    return X, y


def fit_report(C: float) -> dict[str, object]:
    X, y = c_tradeoff_data()
    model = smo.fit_linear_svm_smo(
        X, y, C=C, tolerance=1e-6, max_passes=30, max_iterations=5000
    )
    return soft.analyze_soft_margin_solution(
        smo.linear_weights(model),
        y,
        smo.decision_function(model, X),
        model["alphas"],
        C,
        tolerance=1e-5,
    )


def test_larger_C_trades_larger_weight_norm_for_less_slack() -> None:
    small = fit_report(0.05)
    large = fit_report(10.0)
    assert large["regularization"] > small["regularization"]
    assert np.sum(large["slacks"]) < np.sum(small["slacks"])


def test_trained_solutions_satisfy_kkt_classification() -> None:
    for C in (0.05, 0.2, 10.0):
        assert np.all(fit_report(C)["kkt_ok"])


@pytest.mark.parametrize(
    ("y", "scores"),
    [
        (np.array([1, -1]), np.array([[1.0], [-1.0]])),
        (np.array([[1], [-1]]), np.array([1.0, -1.0])),
        (np.array([0, 1]), np.array([1.0, -1.0])),
        (np.array([1, -1]), np.array([1.0, np.nan])),
    ],
)
def test_signed_margins_reject_invalid_shapes_labels_and_values(y, scores) -> None:
    with pytest.raises(ValueError):
        soft.signed_margins(y, scores)


@pytest.mark.parametrize("C", [0.0, -1.0, np.inf, True])
def test_alpha_status_rejects_invalid_C(C) -> None:
    with pytest.raises(ValueError):
        soft.alpha_status_labels(np.array([0.0, 0.5]), C)


def test_alpha_status_rejects_values_outside_box() -> None:
    with pytest.raises(ValueError):
        soft.alpha_status_labels(np.array([-0.1, 0.5]), 1.0)
    with pytest.raises(ValueError):
        soft.alpha_status_labels(np.array([0.5, 1.1]), 1.0)


def test_overlapping_tolerances_are_rejected() -> None:
    with pytest.raises(ValueError, match="小于0.5"):
        soft.margin_region_labels(np.array([1.0]), tolerance=0.5)
    with pytest.raises(ValueError, match="C/2"):
        soft.alpha_status_labels(np.array([0.0, 0.1]), 0.1, tolerance=0.05)


def test_analysis_rejects_alpha_sample_mismatch() -> None:
    with pytest.raises(ValueError):
        soft.analyze_soft_margin_solution(
            np.array([1.0]),
            np.array([-1, 1]),
            np.array([-1.0, 1.0]),
            np.array([0.5]),
            1.0,
        )


def test_guided_demo_runs_and_reports_three_C_values() -> None:
    demo = CHAPTER / "03_soft_margin" / "guided_demo.py"
    result = subprocess.run(
        [sys.executable, str(demo)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "C=0.05" in result.stdout
    assert "C=0.2" in result.stdout
    assert "C=10" in result.stdout
    assert "regularization/slack penalty/objective" in result.stdout
