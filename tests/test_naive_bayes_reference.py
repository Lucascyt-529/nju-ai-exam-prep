import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "02_machine_learning" / "07_naive_bayes"
spec = importlib.util.spec_from_file_location("naive_bayes_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def categorical_data():
    X = np.array([["sunny", "hot"], ["sunny", "mild"], ["rainy", "mild"], ["rainy", "cool"]])
    return X, np.array([0, 0, 1, 1])


def test_categorical_model_shapes_and_smoothed_priors() -> None:
    X, y = categorical_data(); model = solution.fit_categorical_nb(X, y)
    assert model["class_log_prior"].shape == (2,)
    assert len(model["categories"]) == 2
    assert model["feature_log_prob"][0].shape == (2, 2)
    np.testing.assert_allclose(np.exp(model["class_log_prior"]), [0.5, 0.5])


def test_categorical_probabilities_include_unknown_bucket() -> None:
    X, y = categorical_data(); model = solution.fit_categorical_nb(X, y)
    for feature in range(2):
        known = np.exp(model["feature_log_prob"][feature]).sum(axis=1)
        unknown = np.exp(model["unknown_log_prob"][feature])
        np.testing.assert_allclose(known + unknown, 1.0)


def test_categorical_training_predictions_and_noncontiguous_labels() -> None:
    X, y = categorical_data(); labels = np.where(y == 0, 10, 30)
    model = solution.fit_categorical_nb(X, labels)
    np.testing.assert_array_equal(solution.predict(model, X), labels)


def test_unseen_value_has_finite_scores_and_normalized_posterior() -> None:
    X, y = categorical_data(); model = solution.fit_categorical_nb(X, y)
    query = np.array([["cloudy", "mild"]])
    assert np.all(np.isfinite(solution.joint_log_scores(model, query)))
    posterior = solution.predict_proba(model, query)
    assert posterior.shape == (1, 2)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)


def test_log_score_matches_manual_sum() -> None:
    X, y = categorical_data(); model = solution.fit_categorical_nb(X, y)
    score = solution.joint_log_scores(model, X[:1])[0, 0]
    expected = model["class_log_prior"][0]
    expected += model["feature_log_prob"][0][0, np.flatnonzero(model["categories"][0] == "sunny")[0]]
    expected += model["feature_log_prob"][1][0, np.flatnonzero(model["categories"][1] == "hot")[0]]
    assert score == pytest.approx(expected)


def gaussian_data():
    X = np.array([[0.0, 5.0], [0.2, 5.0], [2.0, 5.0], [2.2, 5.0]])
    return X, np.array([0, 0, 1, 1])


def test_gaussian_means_variances_and_floor_shapes() -> None:
    X, y = gaussian_data(); model = solution.fit_gaussian_nb(X, y, variance_floor=1e-4)
    assert model["means"].shape == (2, 2) and model["variances"].shape == (2, 2)
    np.testing.assert_allclose(model["means"], [[0.1, 5.0], [2.1, 5.0]])
    np.testing.assert_allclose(model["variances"][:, 0], 0.01)
    np.testing.assert_allclose(model["variances"][:, 1], 1e-4)


def test_gaussian_predictions_and_posterior_are_stable() -> None:
    X, y = gaussian_data(); model = solution.fit_gaussian_nb(X, y, variance_floor=1e-4)
    query = np.array([[0.1, 5.0], [2.1, 5.0]])
    np.testing.assert_array_equal(solution.predict(model, query), [0, 1])
    posterior = solution.predict_proba(model, query)
    assert np.all(np.isfinite(posterior)); np.testing.assert_allclose(posterior.sum(axis=1), 1.0)


def test_gaussian_log_score_matches_density_formula() -> None:
    X, y = gaussian_data(); model = solution.fit_gaussian_nb(X, y, variance_floor=1e-4)
    query = np.array([[0.1, 5.0]])
    score = solution.joint_log_scores(model, query)[0, 0]
    expected = model["class_log_prior"][0] - 0.5 * np.sum(np.log(2*np.pi*model["variances"][0]))
    assert score == pytest.approx(expected)


def test_ties_choose_earliest_sorted_class() -> None:
    X = np.array([["same"], ["same"]]); y = np.array([30, 10])
    model = solution.fit_categorical_nb(X, y)
    assert solution.predict(model, np.array([["same"]]))[0] == 10


def test_training_does_not_modify_inputs() -> None:
    X, y = gaussian_data(); X0, y0 = X.copy(), y.copy()
    solution.fit_gaussian_nb(X, y)
    np.testing.assert_array_equal(X, X0); np.testing.assert_array_equal(y, y0)


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf, True])
def test_categorical_rejects_invalid_alpha(alpha) -> None:
    X, y = categorical_data()
    with pytest.raises(ValueError): solution.fit_categorical_nb(X, y, alpha=alpha)


def test_training_rejects_bad_shapes_single_class_and_missing_values() -> None:
    X, y = categorical_data()
    with pytest.raises(ValueError): solution.fit_categorical_nb(X.ravel(), y)
    with pytest.raises(ValueError): solution.fit_categorical_nb(X, np.zeros(4, dtype=int))
    bad = X.astype(object); bad[0, 0] = None
    with pytest.raises(ValueError): solution.fit_categorical_nb(bad, y)
    with pytest.raises(ValueError): solution.fit_gaussian_nb(np.array([[1.0], [np.nan]]), np.array([0, 1]))


@pytest.mark.parametrize("floor", [0.0, -1.0, np.inf, True])
def test_gaussian_rejects_invalid_variance_floor(floor) -> None:
    X, y = gaussian_data()
    with pytest.raises(ValueError): solution.fit_gaussian_nb(X, y, variance_floor=floor)


def test_prediction_rejects_wrong_feature_count_and_wrong_type() -> None:
    X, y = gaussian_data(); model = solution.fit_gaussian_nb(X, y)
    with pytest.raises(ValueError): solution.predict(model, np.ones((2, 3)))
    with pytest.raises(ValueError): solution.predict(model, np.array([["bad", "data"]]))


def test_guided_demo_runs_and_reports_shapes_unknown_and_floor() -> None:
    result = subprocess.run([sys.executable, str(TOPIC / "reference_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "categorical scores shape: (1, 2)" in result.stdout
    assert "unseen-value posterior:" in result.stdout
    assert "gaussian means shape: (2, 2)" in result.stdout
    assert "gaussian prediction: [1]" in result.stdout
