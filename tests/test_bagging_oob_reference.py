import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "08_ensemble_learning" / "02_bagging_oob"
spec = importlib.util.spec_from_file_location("bagging_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def data():
    return np.arange(8.0).reshape(-1, 1), np.array([-1, -1, 1, -1, 1, 1, -1, 1])


def test_bootstrap_shape_range_repetition_and_reproducibility() -> None:
    first = solution.bootstrap_sample_indices(8, 20, random_state=7)
    second = solution.bootstrap_sample_indices(8, 20, random_state=7)
    assert first.shape == (20, 8) and np.all((first >= 0) & (first < 8))
    np.testing.assert_array_equal(first, second)
    assert np.unique(first[0]).size < 8
    np.testing.assert_array_equal(first[0], [7, 5, 5, 7, 4, 6, 6, 1])


def test_different_seed_changes_bootstrap_matrix() -> None:
    assert not np.array_equal(
        solution.bootstrap_sample_indices(8, 5, random_state=1),
        solution.bootstrap_sample_indices(8, 5, random_state=2),
    )


def test_oob_indices_are_sorted_complement_of_unique_inbag() -> None:
    indices = np.array([7, 5, 5, 7, 4, 6, 6, 1])
    np.testing.assert_array_equal(solution.out_of_bag_indices(indices, 8), [0, 2, 3])


def test_bagging_model_shapes_and_learner_count() -> None:
    X, y = data(); model = solution.fit_bagging_stumps(X, y, n_estimators=20, random_state=7)
    assert len(model["learners"]) == 20
    assert model["bootstrap_indices"].shape == (20, 8)
    assert model["n_train"] == 8 and model["n_features"] == 1


def test_base_predictions_scores_and_labels_have_expected_shapes() -> None:
    X, y = data(); model = solution.fit_bagging_stumps(X, y, n_estimators=10, random_state=3)
    assert solution.base_predictions(model, X).shape == (10, 8)
    assert solution.decision_function(model, X).shape == (8,)
    assert set(np.unique(solution.predict(model, X))).issubset({-1, 1})


def test_hard_vote_tie_is_positive() -> None:
    model = {"learners": ({"kind": "constant", "label": -1}, {"kind": "constant", "label": 1}),
             "bootstrap_indices": np.array([[0, 0], [1, 1]]), "n_features": 1, "n_train": 2}
    X = np.array([[0.0], [1.0]])
    np.testing.assert_array_equal(solution.decision_function(model, X), [0.0, 0.0])
    np.testing.assert_array_equal(solution.predict(model, X), [1, 1])


def test_oob_counts_match_manual_membership() -> None:
    model = {"learners": ({"kind": "constant", "label": -1}, {"kind": "constant", "label": 1}),
             "bootstrap_indices": np.array([[0, 0, 1], [1, 2, 2]]), "n_features": 1, "n_train": 3}
    X = np.arange(3.0).reshape(-1, 1)
    scores, counts = solution.oob_decision_function(model, X)
    np.testing.assert_array_equal(counts, [1, 0, 1])
    assert scores[0] == pytest.approx(1.0) and np.isnan(scores[1]) and scores[2] == pytest.approx(-1.0)


def test_oob_accuracy_only_uses_covered_samples() -> None:
    model = {"learners": ({"kind": "constant", "label": -1}, {"kind": "constant", "label": 1}),
             "bootstrap_indices": np.array([[0, 0, 1], [1, 2, 2]]), "n_features": 1, "n_train": 3}
    X = np.arange(3.0).reshape(-1, 1); y = np.array([1, -1, -1])
    accuracy, covered = solution.oob_accuracy(model, X, y)
    np.testing.assert_array_equal(covered, [True, False, True])
    assert accuracy == pytest.approx(1.0)


def test_no_oob_coverage_is_explicit_error() -> None:
    model = {"learners": ({"kind": "constant", "label": 1},),
             "bootstrap_indices": np.array([[0, 1]]), "n_features": 1, "n_train": 2}
    with pytest.raises(ValueError, match="没有任何"):
        solution.oob_accuracy(model, np.array([[0.0], [1.0]]), np.array([-1, 1]))


def test_more_estimators_cover_all_samples_in_fixed_example() -> None:
    X, y = data(); model = solution.fit_bagging_stumps(X, y, n_estimators=20, random_state=7)
    _, counts = solution.oob_decision_function(model, X)
    assert np.all(counts > 0)


def test_fit_is_deterministic_and_does_not_modify_inputs() -> None:
    X, y = data(); X0, y0 = X.copy(), y.copy()
    first = solution.fit_bagging_stumps(X, y, random_state=4); second = solution.fit_bagging_stumps(X, y, random_state=4)
    assert first["learners"] == second["learners"]
    np.testing.assert_array_equal(first["bootstrap_indices"], second["bootstrap_indices"])
    np.testing.assert_array_equal(X, X0); np.testing.assert_array_equal(y, y0)


@pytest.mark.parametrize("args", [(0, 2), (3, 0), (-1, 2), (3.5, 2), (True, 2)])
def test_bootstrap_rejects_invalid_sizes(args) -> None:
    with pytest.raises(ValueError): solution.bootstrap_sample_indices(*args)


def test_oob_rejects_noninteger_and_out_of_range_indices() -> None:
    with pytest.raises(ValueError): solution.out_of_bag_indices(np.array([0.0, 1.0]), 2)
    with pytest.raises(ValueError): solution.out_of_bag_indices(np.array([0, 2]), 2)


def test_fit_and_prediction_reject_bad_shapes_labels_and_feature_count() -> None:
    X, y = data()
    with pytest.raises(ValueError): solution.fit_bagging_stumps(X.ravel(), y)
    with pytest.raises(ValueError): solution.fit_bagging_stumps(X, np.where(y == -1, 0, 1))
    model = solution.fit_bagging_stumps(X, y)
    with pytest.raises(ValueError): solution.predict(model, np.ones((2, 2)))
    with pytest.raises(ValueError): solution.oob_decision_function(model, X[:-1])


def test_guided_demo_runs_and_reports_bootstrap_oob_and_coverage() -> None:
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "bootstrap shape: (20, 8)" in result.stdout
    assert "first OOB: [0, 2, 3]" in result.stdout
    assert "covered: 8 / 8" in result.stdout
    assert "OOB accuracy:" in result.stdout

