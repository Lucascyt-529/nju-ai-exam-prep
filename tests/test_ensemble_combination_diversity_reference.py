import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "08_ensemble_learning" / "04_combination_diversity"
spec = importlib.util.spec_from_file_location("ensemble_diversity_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def test_equal_and_weighted_regression_average() -> None:
    predictions = np.array([[1.,2.,3.],[3.,4.,5.]])
    np.testing.assert_allclose(solution.weighted_average(predictions), [2,3,4])
    np.testing.assert_allclose(solution.weighted_average(predictions, np.array([0.25,0.75])), [2.5,3.5,4.5])


def test_hard_vote_and_sorted_label_tie_rule() -> None:
    predictions = np.array([[1,10,30],[1,30,10],[0,10,30],[0,30,10]])
    np.testing.assert_array_equal(solution.hard_vote(predictions), [0,10,10])


def test_soft_vote_matches_weighted_probability_average_and_tie() -> None:
    probabilities = np.array([[[.8,.2],[.5,.5]], [[.2,.8],[.5,.5]]])
    combined, labels = solution.soft_vote(probabilities, np.array([.75,.25]))
    np.testing.assert_allclose(combined, [[.65,.35],[.5,.5]])
    np.testing.assert_array_equal(labels, [0,0])


def test_pairwise_contingency_counts_all_four_cases() -> None:
    y=np.array([1,1,-1,-1,1,-1]); a=np.array([1,1,-1,1,-1,-1]); b=np.array([1,-1,-1,-1,1,1])
    assert solution.pairwise_contingency(y,a,b) == {"n11":2,"n00":0,"n10":2,"n01":2}


def test_diversity_metrics_match_hand_formulas() -> None:
    counts={"n11":4,"n00":2,"n10":1,"n01":3}; metrics=solution.diversity_metrics(counts)
    assert metrics["q"] == pytest.approx((8-3)/(8+3))
    assert metrics["disagreement"] == pytest.approx(.4)
    assert metrics["double_fault"] == pytest.approx(.2)
    expected_corr=(8-3)/np.sqrt(5*5*7*3)
    assert metrics["correlation"] == pytest.approx(expected_corr)


def test_q_and_correlation_undefined_boundaries_return_nan() -> None:
    metrics=solution.diversity_metrics({"n11":5,"n00":0,"n10":0,"n01":0})
    assert np.isnan(metrics["q"]) and np.isnan(metrics["correlation"])
    assert metrics["disagreement"] == 0 and metrics["double_fault"] == 0


def test_identical_and_opposite_correctness_have_expected_diversity() -> None:
    identical=solution.diversity_metrics({"n11":3,"n00":2,"n10":0,"n01":0})
    opposite=solution.diversity_metrics({"n11":0,"n00":0,"n10":3,"n01":2})
    assert identical["q"]==pytest.approx(1) and identical["disagreement"]==0
    assert opposite["q"]==pytest.approx(-1) and opposite["disagreement"]==1


def test_regression_error_ambiguity_identity_holds_per_sample() -> None:
    y=np.array([1.,2.,3.]); predictions=np.array([[1.,2.5,2.5],[0.,2.,4.],[2.,1.5,3.]])
    report=solution.regression_error_ambiguity(y,predictions)
    np.testing.assert_allclose(report["ensemble_error"], report["mean_individual_error"]-report["ambiguity"])
    np.testing.assert_allclose(report["ensemble_prediction"], predictions.mean(axis=0))


def test_identical_regressors_have_zero_ambiguity() -> None:
    predictions=np.array([[1.,2.],[1.,2.],[1.,2.]])
    report=solution.regression_error_ambiguity(np.array([0.,0.]),predictions)
    np.testing.assert_allclose(report["ambiguity"],0)
    np.testing.assert_allclose(report["ensemble_error"],report["mean_individual_error"])


def test_inputs_are_not_modified() -> None:
    predictions=np.array([[1.,2.],[3.,4.]]); weights=np.array([.4,.6]); p0=predictions.copy(); w0=weights.copy()
    solution.weighted_average(predictions,weights)
    np.testing.assert_array_equal(predictions,p0); np.testing.assert_array_equal(weights,w0)


@pytest.mark.parametrize("weights", [np.array([1.]),np.array([.2,.2]),np.array([-.1,1.1]),np.array([np.nan,.0])])
def test_weighted_average_rejects_bad_weights(weights) -> None:
    with pytest.raises(ValueError): solution.weighted_average(np.ones((2,3)),weights)


def test_soft_vote_rejects_bad_shape_negative_and_non_normalized_probabilities() -> None:
    for bad in (np.ones((2,3)), np.array([[[-.1,1.1]]]), np.array([[[.2,.2]]])):
        with pytest.raises(ValueError): solution.soft_vote(bad)


def test_contingency_rejects_shape_and_nonfinite_mismatch() -> None:
    y=np.array([1.,-1.])
    with pytest.raises(ValueError): solution.pairwise_contingency(y,np.array([1.]),np.array([1.,-1.]))
    with pytest.raises(ValueError): solution.pairwise_contingency(y,np.array([1.,np.nan]),np.array([1.,-1.]))


def test_diversity_rejects_bad_keys_negative_noninteger_and_zero_total() -> None:
    for bad in ({"n11":1}, {"n11":-1,"n00":1,"n10":0,"n01":0}, {"n11":1.5,"n00":1,"n10":0,"n01":0}, {"n11":0,"n00":0,"n10":0,"n01":0}):
        with pytest.raises(ValueError): solution.diversity_metrics(bad)


def test_regression_decomposition_rejects_sample_mismatch() -> None:
    with pytest.raises(ValueError): solution.regression_error_ambiguity(np.array([1.,2.]),np.ones((3,3)))


def test_guided_demo_runs_and_reports_identity() -> None:
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "contingency: {'n11': 2, 'n00': 0, 'n10': 2, 'n01': 2}" in result.stdout
    assert "metrics:" in result.stdout
    assert "identity holds: True" in result.stdout

