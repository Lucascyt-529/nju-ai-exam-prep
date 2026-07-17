import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "08_ensemble_learning" / "03_random_forest"
spec = importlib.util.spec_from_file_location("random_forest_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def data():
    x = np.arange(12.0); X = np.column_stack((x, x[::-1], np.sin(x), np.cos(x)))
    y = np.where((x < 3) | ((x >= 6) & (x < 9)), -1, 1)
    return X, y


def test_feature_subsets_shape_range_unique_sorted_and_reproducible() -> None:
    first = solution.sample_feature_subsets(5, 20, 2, random_state=4)
    second = solution.sample_feature_subsets(5, 20, 2, random_state=4)
    assert first.shape == (20, 2); np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0) & (first < 5))
    assert all(np.unique(row).size == 2 and np.all(np.diff(row) > 0) for row in first)


def test_full_feature_subsets_are_all_features() -> None:
    subsets = solution.sample_feature_subsets(4, 5, 4, random_state=9)
    np.testing.assert_array_equal(subsets, np.tile(np.arange(4), (5, 1)))


def test_different_seed_changes_feature_subsets() -> None:
    assert not np.array_equal(solution.sample_feature_subsets(6, 10, 2, random_state=1), solution.sample_feature_subsets(6, 10, 2, random_state=2))


def test_forest_model_shapes_and_each_stump_respects_candidate_subset() -> None:
    X, y = data(); model = solution.fit_random_subspace_forest(X, y, n_estimators=30, max_features=2, random_state=5)
    assert model["bootstrap_indices"].shape == (30, 12)
    assert model["feature_subsets"].shape == (30, 2)
    for learner, subset in zip(model["learners"], model["feature_subsets"]):
        if learner["kind"] == "stump": assert learner["feature"] in subset


def test_random_subspaces_use_multiple_candidate_sets_and_features() -> None:
    X, y = data(); model = solution.fit_random_subspace_forest(X, y, n_estimators=40, max_features=2, random_state=5)
    assert np.unique(model["feature_subsets"], axis=0).shape[0] > 1
    chosen = {learner["feature"] for learner in model["learners"] if learner["kind"] == "stump"}
    assert len(chosen) > 1


def test_prediction_matrix_scores_and_labels_shapes() -> None:
    X, y = data(); model = solution.fit_random_subspace_forest(X, y, n_estimators=12, max_features=2, random_state=5)
    assert solution.base_predictions(model, X).shape == (12, 12)
    assert solution.decision_function(model, X).shape == (12,)
    assert set(np.unique(solution.predict(model, X))).issubset({-1, 1})


def test_vote_tie_is_positive() -> None:
    model = {"learners": ({"kind": "constant", "label": -1}, {"kind": "constant", "label": 1}),
             "bootstrap_indices": np.array([[0], [0]]), "feature_subsets": np.array([[0], [0]]), "n_features": 1, "max_features": 1}
    X = np.array([[0.0]])
    assert solution.decision_function(model, X)[0] == 0 and solution.predict(model, X)[0] == 1


def test_pairwise_prediction_correlation_matches_known_pairs() -> None:
    a = np.array([1, -1, 1, -1]); b = a.copy(); c = -a
    assert solution.mean_pairwise_prediction_correlation(np.vstack((a, b))) == pytest.approx(1.0)
    assert solution.mean_pairwise_prediction_correlation(np.vstack((a, c))) == pytest.approx(-1.0)
    assert solution.mean_pairwise_prediction_correlation(np.vstack((a, b, c))) == pytest.approx(-1/3)


def test_constant_prediction_pairs_are_excluded_or_nan() -> None:
    varying = np.array([1, -1, 1, -1]); constant = np.ones(4)
    assert np.isnan(solution.mean_pairwise_prediction_correlation(np.vstack((constant, constant))))
    value = solution.mean_pairwise_prediction_correlation(np.vstack((varying, varying, constant)))
    assert value == pytest.approx(1.0)


def test_fit_is_deterministic_and_does_not_modify_inputs() -> None:
    X, y = data(); X0, y0 = X.copy(), y.copy()
    first = solution.fit_random_subspace_forest(X, y, n_estimators=15, max_features=2, random_state=8)
    second = solution.fit_random_subspace_forest(X, y, n_estimators=15, max_features=2, random_state=8)
    assert first["learners"] == second["learners"]
    np.testing.assert_array_equal(first["bootstrap_indices"], second["bootstrap_indices"])
    np.testing.assert_array_equal(first["feature_subsets"], second["feature_subsets"])
    np.testing.assert_array_equal(X, X0); np.testing.assert_array_equal(y, y0)


@pytest.mark.parametrize("args", [(0, 2, 1), (4, 0, 1), (4, 2, 0), (4, 2, 5), (4.5, 2, 1), (True, 2, 1)])
def test_feature_sampling_rejects_invalid_sizes(args) -> None:
    with pytest.raises(ValueError): solution.sample_feature_subsets(*args)


def test_fit_rejects_bad_shapes_labels_max_features_and_query() -> None:
    X, y = data()
    with pytest.raises(ValueError): solution.fit_random_subspace_forest(X.ravel(), y)
    with pytest.raises(ValueError): solution.fit_random_subspace_forest(X, np.where(y == -1, 0, 1))
    with pytest.raises(ValueError): solution.fit_random_subspace_forest(X, y, max_features=5)
    model = solution.fit_random_subspace_forest(X, y, max_features=2)
    with pytest.raises(ValueError): solution.predict(model, np.ones((2, 3)))


def test_correlation_rejects_bad_prediction_matrices() -> None:
    for bad in (np.array([1, -1]), np.ones((1, 3)), np.array([[1, np.nan], [1, -1]])):
        with pytest.raises(ValueError): solution.mean_pairwise_prediction_correlation(bad)


def test_guided_demo_runs_and_reports_subsets_features_and_correlation() -> None:
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "feature subsets shape: (12, 2)" in result.stdout
    assert "chosen features:" in result.stdout
    assert "mean prediction correlation:" in result.stdout
    assert "ensemble prediction:" in result.stdout

