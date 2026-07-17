import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"07_bayesian_classifiers"/"06_em"
spec=importlib.util.spec_from_file_location("em_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def data(): return np.array([1,2,1,8,9,8]),np.full(6,10)
def test_symmetric_parameters_give_equal_responsibility():
    h,n=data(); r=solution.expectation_step(h,n,np.array([.5,.5]),np.array([.4,.4])); np.testing.assert_allclose(r,.5)
def test_responsibility_shape_rows_sum_and_extreme_assignment():
    h,n=data(); r=solution.expectation_step(h,n,np.array([.5,.5]),np.array([.2,.8])); assert r.shape==(6,2); np.testing.assert_allclose(r.sum(axis=1),1); assert r[0,0]>.99 and r[-1,1]>.99
def test_m_step_matches_soft_count_formula():
    h=np.array([2,8]); n=np.array([10,10]); r=np.array([[.75,.25],[.25,.75]]); mixing,p=solution.maximization_step(h,n,r); np.testing.assert_allclose(mixing,[.5,.5]); np.testing.assert_allclose(p,[.35,.65])
def test_likelihood_includes_binomial_and_is_finite():
    h,n=data(); value=solution.observed_log_likelihood(h,n,np.array([.5,.5]),np.array([.2,.8])); assert np.isfinite(value)
def test_em_likelihood_nondecreasing_and_recovers_two_groups():
    h,n=data(); model=solution.fit_coin_mixture_em(h,n,initial_probabilities=np.array([.25,.75])); assert np.all(np.diff(model["log_likelihood_history"])>=-1e-10); assert model["probabilities"][0]<.25 and model["probabilities"][1]>.75; assert model["mixing"].sum()==pytest.approx(1)
def test_final_responsibilities_match_final_parameters():
    h,n=data(); model=solution.fit_coin_mixture_em(h,n); expected=solution.expectation_step(h,n,model["mixing"],model["probabilities"]); np.testing.assert_allclose(model["responsibilities"],expected)
def test_label_swapped_initialization_swaps_solution():
    h,n=data(); a=solution.fit_coin_mixture_em(h,n,initial_probabilities=np.array([.2,.8])); b=solution.fit_coin_mixture_em(h,n,initial_probabilities=np.array([.8,.2])); np.testing.assert_allclose(a["probabilities"],b["probabilities"][::-1],atol=1e-7)
def test_repeatable_and_inputs_unmodified():
    h,n=data(); h0=h.copy(); a=solution.fit_coin_mixture_em(h,n); b=solution.fit_coin_mixture_em(h,n); np.testing.assert_allclose(a["log_likelihood_history"],b["log_likelihood_history"]); np.testing.assert_array_equal(h,h0)
def test_single_iteration_respected():
    h,n=data(); model=solution.fit_coin_mixture_em(h,n,max_iterations=1); assert model["iterations"]==1 and len(model["log_likelihood_history"])==2
def test_bad_counts_parameters_and_responsibilities_rejected():
    h,n=data()
    with pytest.raises(ValueError): solution.expectation_step(np.array([11]),np.array([10]),np.array([.5,.5]),np.array([.2,.8]))
    with pytest.raises(ValueError): solution.expectation_step(h,n,np.array([.2,.2]),np.array([.2,.8]))
    with pytest.raises(ValueError): solution.maximization_step(h,n,np.ones((6,2)))
@pytest.mark.parametrize(("name","value"),[("max_iterations",0),("tolerance",0)])
def test_bad_fit_options_rejected(name,value):
    h,n=data()
    with pytest.raises(ValueError): solution.fit_coin_mixture_em(h,n,**{name:value})
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "responsibility shape: (6, 2)" in result.stdout and "likelihood nondecreasing: True" in result.stdout
