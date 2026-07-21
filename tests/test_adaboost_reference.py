import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "02_machine_learning" / "09_ensemble_learning" / "01_adaboost"
spec = importlib.util.spec_from_file_location("adaboost_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def data():
    return np.arange(5.0).reshape(-1, 1), np.array([-1, 1, -1, 1, 1])


def test_candidate_thresholds_include_constants_and_unique_midpoints() -> None:
    thresholds = solution.candidate_thresholds(np.array([0.0, 1.0, 1.0, 3.0]))
    assert thresholds.size == 4
    assert thresholds[0] < 0 and thresholds[-1] > 3
    np.testing.assert_allclose(thresholds[1:-1], [0.5, 2.0])


def test_stump_prediction_polarity_and_zero_boundary() -> None:
    X = np.array([[-1.0], [0.0], [1.0]])
    np.testing.assert_array_equal(solution.stump_predict(X, 0, 0.0, 1), [-1, 1, 1])
    np.testing.assert_array_equal(solution.stump_predict(X, 0, 0.0, -1), [1, -1, -1])


def test_weighted_stump_finds_known_first_split() -> None:
    X, y = data(); weights = np.full(5, 0.2)
    stump = solution.fit_weighted_stump(X, y, weights)
    assert stump == {"feature": 0, "threshold": 0.5, "polarity": 1, "error": pytest.approx(0.2)}


def test_feature_tie_prefers_earlier_feature() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([-1, 1, 1]); stump = solution.fit_weighted_stump(X, y, np.full(3, 1/3))
    assert stump["feature"] == 0


def test_classifier_weight_matches_hand_formula() -> None:
    assert solution.classifier_weight(0.2) == pytest.approx(0.5 * np.log(4.0))
    assert np.isfinite(solution.classifier_weight(0.0))


def test_weight_update_matches_formula_and_normalizes() -> None:
    weights = np.full(4, 0.25); y = np.array([-1, -1, 1, 1]); prediction = np.array([-1, 1, 1, 1])
    alpha = solution.classifier_weight(0.25)
    updated = solution.update_sample_weights(weights, y, prediction, alpha)
    raw = weights * np.exp(-alpha * y * prediction)
    np.testing.assert_allclose(updated, raw / raw.sum())
    assert updated.sum() == pytest.approx(1.0)


def test_misclassified_sample_relative_weight_increases() -> None:
    weights = np.full(4, 0.25); y = np.array([-1, -1, 1, 1]); prediction = np.array([-1, 1, 1, 1])
    updated = solution.update_sample_weights(weights, y, prediction, solution.classifier_weight(0.25))
    assert updated[1] > updated[0] and updated[1] > updated[2]


def test_adaboost_fits_pattern_requiring_multiple_stumps() -> None:
    X, y = data(); model = solution.fit_adaboost(X, y, n_estimators=10)
    assert len(model["learners"]) > 1
    np.testing.assert_array_equal(solution.predict(model, X), y)
    assert model["weight_history"].shape == (len(model["learners"]) + 1, 5)
    np.testing.assert_allclose(model["weight_history"].sum(axis=1), 1.0)


def test_exponential_training_loss_decreases_across_rounds() -> None:
    X, y = data(); model = solution.fit_adaboost(X, y, n_estimators=6)
    scores = np.zeros(len(y)); losses = [np.mean(np.exp(-y * scores))]
    for learner in model["learners"]:
        scores += learner["alpha"] * solution.stump_predict(X, learner["feature"], learner["threshold"], learner["polarity"])
        losses.append(np.mean(np.exp(-y * scores)))
    assert np.all(np.diff(losses) < 0)


def test_perfect_stump_stops_after_one_round() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]]); y = np.array([-1, -1, 1, 1])
    model = solution.fit_adaboost(X, y, n_estimators=10)
    assert len(model["learners"]) == 1
    np.testing.assert_array_equal(solution.predict(model, X), y)


def test_random_quality_stump_adds_no_learner() -> None:
    X = np.zeros((4, 1)); y = np.array([-1, 1, -1, 1])
    model = solution.fit_adaboost(X, y, n_estimators=5)
    assert len(model["learners"]) == 0
    np.testing.assert_array_equal(solution.predict(model, X), np.ones(4, dtype=int))


def test_fit_is_deterministic_and_does_not_modify_inputs() -> None:
    X, y = data(); X0, y0 = X.copy(), y.copy()
    first = solution.fit_adaboost(X, y); second = solution.fit_adaboost(X, y)
    assert first["learners"] == second["learners"]
    np.testing.assert_array_equal(first["weight_history"], second["weight_history"])
    np.testing.assert_array_equal(X, X0); np.testing.assert_array_equal(y, y0)


@pytest.mark.parametrize("error", [-0.1, 0.5, 0.7, np.inf, True])
def test_classifier_weight_rejects_invalid_error(error) -> None:
    with pytest.raises(ValueError): solution.classifier_weight(error)


def test_training_rejects_bad_shapes_labels_weights_and_estimator_count() -> None:
    X, y = data()
    with pytest.raises(ValueError): solution.fit_adaboost(X.ravel(), y)
    with pytest.raises(ValueError): solution.fit_adaboost(X, np.array([0, 0, 1, 1, 1]))
    with pytest.raises(ValueError): solution.fit_adaboost(X, y, n_estimators=0)
    with pytest.raises(ValueError): solution.fit_weighted_stump(X, y, np.ones(5))


def test_prediction_rejects_feature_mismatch() -> None:
    X, y = data(); model = solution.fit_adaboost(X, y)
    with pytest.raises(ValueError): solution.predict(model, np.ones((2, 2)))


def test_guided_demo_runs_and_reports_weight_shift_and_fit() -> None:
    result = subprocess.run([sys.executable, str(TOPIC / "reference_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "first stump:" in result.stdout
    assert "after first round: [0.125 0.125 0.5   0.125 0.125]" in result.stdout
    assert "training prediction: [-1, 1, -1, 1, 1]" in result.stdout
