import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"05_neural_networks"/"11_boltzmann_machine"
spec=importlib.util.spec_from_file_location("bm_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def params(): return np.array([[0.,1.,-.5],[1.,0.,.8],[-.5,.8,0.]]),np.array([.2,-.1,.3])
def test_state_enumeration_shape_order_unique():
    S=solution.enumerate_binary_states(3); assert S.shape==(8,3); np.testing.assert_array_equal(S[0],[0,0,0]); np.testing.assert_array_equal(S[-1],[1,1,1]); assert len(np.unique(S,axis=0))==8
def test_energy_hand_values_and_shape():
    W,b=params(); S=np.array([[0,0,0],[1,1,0]]); E=solution.energy(S,W,b); np.testing.assert_allclose(E,[0,-1.1]); assert E.shape==(2,)
def test_exact_probabilities_positive_normalized_and_low_energy_higher():
    W,b=params(); model=solution.exact_distribution(W,b); assert np.all(model["probabilities"]>0); assert model["probabilities"].sum()==pytest.approx(1); assert np.argmax(model["probabilities"])==np.argmin(model["energies"])
def test_zero_parameters_uniform_distribution_and_moments():
    W=np.zeros((3,3)); b=np.zeros(3); model=solution.exact_distribution(W,b); np.testing.assert_allclose(model["probabilities"],1/8); np.testing.assert_allclose(model["mean"],.5); np.testing.assert_allclose(np.diag(model["second_moment"]),.5); assert model["second_moment"][0,1]==pytest.approx(.25)
def test_conditional_matches_exact_joint_ratio():
    W,b=params(); model=solution.exact_distribution(W,b); mask=(model["states"][:,1]==1)&(model["states"][:,2]==0); exact=model["probabilities"][mask & (model["states"][:,0]==1)].sum()/model["probabilities"][mask].sum(); direct=solution.conditional_probability_one(np.array([0,1,0]),0,W,b); assert direct==pytest.approx(exact)
def test_positive_field_probability_above_half_and_temperature_flattens():
    W,b=params(); state=np.array([0,1,0]); assert solution.conditional_probability_one(state,0,W,b)>.5
    low=solution.exact_distribution(W,b,temperature=.2)["probabilities"]; high=solution.exact_distribution(W,b,temperature=10)["probabilities"]; assert low.max()-low.min()>high.max()-high.min()
def test_gibbs_shape_initial_row_binary_repeatable_and_no_mutation():
    W,b=params(); initial=np.array([0,0,0]); original=initial.copy(); a=solution.gibbs_sample(initial,W,b,n_sweeps=5,random_state=4); c=solution.gibbs_sample(initial,W,b,n_sweeps=5,random_state=4); assert a.shape==(6,3); np.testing.assert_array_equal(a[0],initial); assert np.all(np.isin(a,[0,1])); np.testing.assert_array_equal(a,c); np.testing.assert_array_equal(initial,original)
def test_zero_sweeps_returns_initial_only():
    W,b=params(); trajectory=solution.gibbs_sample(np.array([1,0,1]),W,b,n_sweeps=0); np.testing.assert_array_equal(trajectory,[[1,0,1]])
@pytest.mark.parametrize("n",[0,-1,21,1.5,True])
def test_bad_unit_count_rejected(n):
    with pytest.raises(ValueError): solution.enumerate_binary_states(n)
def test_bad_weights_biases_states_rejected():
    W,b=params()
    with pytest.raises(ValueError): solution.exact_distribution(np.ones((2,3)),np.zeros(2))
    with pytest.raises(ValueError): solution.exact_distribution(np.array([[0,1],[0,0.]]),np.zeros(2))
    with pytest.raises(ValueError): solution.exact_distribution(np.eye(2),np.zeros(2))
    with pytest.raises(ValueError): solution.energy(np.array([[0,2,0]]),W,b)
def test_bad_temperature_unit_and_sampling_options_rejected():
    W,b=params(); state=np.array([0,0,0])
    with pytest.raises(ValueError): solution.exact_distribution(W,b,temperature=0)
    with pytest.raises(ValueError): solution.conditional_probability_one(state,3,W,b)
    with pytest.raises(ValueError): solution.gibbs_sample(state,W,b,n_sweeps=-1)
    with pytest.raises(ValueError): solution.gibbs_sample(state,W,b,n_sweeps=1,random_state=1.5)
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "states shape: (8, 3)" in result.stdout and "probability sum: 1.0" in result.stdout and "trajectory shape: (6, 3)" in result.stdout
