import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "07_bayesian_classifiers" / "04_aode"
spec = importlib.util.spec_from_file_location("aode_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def data():
    X = np.array([["red", "round", "sweet"], ["red", "round", "sweet"], ["green", "long", "sour"], ["green", "long", "sour"], ["red", "long", "sour"], ["green", "round", "sweet"]])
    return X, np.array([1, 1, 0, 0, 0, 1])


def test_fit_stores_sorted_classes_feature_domains_and_copies() -> None:
    X, y = data(); model = solution.fit_aode(X, y, min_parent_count=2)
    np.testing.assert_array_equal(model["classes"], [0, 1])
    assert len(model["categories"]) == 3 and model["n_features"] == 3
    X[0, 0] = "changed"; y[0] = 9
    assert model["X_train"][0, 0] == "red" and model["y_train"][0] == 1


def test_eligible_parents_follow_global_frequency_threshold() -> None:
    X, y = data(); model = solution.fit_aode(X, y, min_parent_count=3)
    np.testing.assert_array_equal(
        solution.eligible_parent_indices(model, np.array(["red", "round", "sweet"])),
        [0, 1, 2],
    )
    model4 = solution.fit_aode(X, y, min_parent_count=4)
    assert solution.eligible_parent_indices(model4, np.array(["red", "round", "sweet"])).size == 0


def test_unseen_values_are_not_eligible_parents() -> None:
    X, y = data(); model = solution.fit_aode(X, y)
    assert solution.eligible_parent_indices(model, np.array(["blue", "square", "bitter"])).size == 0


def test_no_eligible_parent_falls_back_exactly_to_naive_scores() -> None:
    X, y = data(); model = solution.fit_aode(X, y, min_parent_count=10)
    query = np.array([["red", "round", "sweet"], ["blue", "square", "bitter"]])
    np.testing.assert_allclose(solution.joint_log_scores(model, query), solution.naive_joint_log_scores(model, query))


def test_single_spode_score_matches_hand_count_formula() -> None:
    X, y = data(); model = solution.fit_aode(X, y, alpha=1.0)
    row = np.array(["red", "round", "sweet"])
    scores = solution._spode_scores(model, row, 0)
    # class 1: count(c,red)=2; P(c,red)=3/(6+2*3)
    # round and sweet counts under (c=1,red) are both 2; each conditional=3/(2+3)
    expected = np.log(3/12) + 2*np.log(3/5)
    assert scores[1] == pytest.approx(expected)


def test_aode_uses_dependency_and_differs_from_naive_score() -> None:
    X, y = data(); model = solution.fit_aode(X, y, min_parent_count=2)
    query = np.array([["red", "round", "sweet"]])
    assert not np.allclose(solution.joint_log_scores(model, query), solution.naive_joint_log_scores(model, query))


def test_aode_predicts_training_pattern_and_normalizes_posterior() -> None:
    X, y = data(); model = solution.fit_aode(X, y, min_parent_count=2)
    query = np.array([["red", "round", "sweet"], ["green", "long", "sour"]])
    np.testing.assert_array_equal(solution.predict(model, query), [1, 0])
    posterior = solution.predict_proba(model, query)
    assert posterior.shape == (2, 2) and np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)


def test_log_mean_exp_is_invariant_to_parent_order() -> None:
    X, y = data(); model = solution.fit_aode(X[:, ::-1], y, min_parent_count=2)
    original = solution.fit_aode(X, y, min_parent_count=2)
    query = np.array([["red", "round", "sweet"]])
    np.testing.assert_allclose(
        solution.predict_proba(original, query),
        solution.predict_proba(model, query[:, ::-1]),
    )


def test_noncontiguous_labels_and_deterministic_tie() -> None:
    X = np.array([["same"], ["same"]]); y = np.array([30, 10])
    model = solution.fit_aode(X, y)
    assert solution.predict(model, np.array([["same"]]))[0] == 10


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf, True])
def test_fit_rejects_invalid_alpha(alpha) -> None:
    X, y = data()
    with pytest.raises(ValueError): solution.fit_aode(X, y, alpha=alpha)


@pytest.mark.parametrize("threshold", [0, -1, 1.5, True])
def test_fit_rejects_invalid_parent_threshold(threshold) -> None:
    X, y = data()
    with pytest.raises(ValueError): solution.fit_aode(X, y, min_parent_count=threshold)


def test_fit_and_prediction_reject_bad_shapes_missing_and_feature_mismatch() -> None:
    X, y = data()
    with pytest.raises(ValueError): solution.fit_aode(X.ravel(), y)
    with pytest.raises(ValueError): solution.fit_aode(X, np.zeros(6, dtype=int))
    bad = X.astype(object); bad[0, 0] = None
    with pytest.raises(ValueError): solution.fit_aode(bad, y)
    model = solution.fit_aode(X, y)
    with pytest.raises(ValueError): solution.joint_log_scores(model, np.ones((2, 4)))
    with pytest.raises(ValueError): solution.eligible_parent_indices(model, np.array(["red"]))


def test_guided_demo_runs_and_shows_parent_selection_and_fallback() -> None:
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "eligible parents first: [0, 1, 2]" in result.stdout
    assert "eligible parents unseen: []" in result.stdout
    assert "naive fallback unseen:" in result.stdout
    assert "posterior row sums: [1. 1.]" in result.stdout

