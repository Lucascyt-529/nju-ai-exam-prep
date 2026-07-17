import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"01_introduction"/"01_hypothesis_preference_nfl"
spec=importlib.util.spec_from_file_location("intro_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def test_hypothesis_count_shape_order_and_uniqueness():
    H=solution.enumerate_binary_hypotheses(3); assert H.shape==(8,3); np.testing.assert_array_equal(H[0],[0,0,0]); np.testing.assert_array_equal(H[-1],[1,1,1]); assert len(np.unique(H,axis=0))==8
def test_version_space_size_and_consistency():
    H=solution.enumerate_binary_hypotheses(4); V=solution.version_space(H,np.array([0,3]),np.array([0,1])); assert V.shape==(4,4); assert np.all(V[:,0]==0) and np.all(V[:,3]==1)
def test_all_points_observed_leaves_one_hypothesis():
    H=solution.enumerate_binary_hypotheses(3); V=solution.version_space(H,np.arange(3),np.array([1,0,1])); np.testing.assert_array_equal(V,[[1,0,1]])
@pytest.mark.parametrize(("h","count"),[([0,0,0],0),([0,1,1],1),([0,1,0,1],3)])
def test_transition_count(h,count): assert solution.transition_count(np.array(h))==count
def test_preference_minimizes_transitions_then_lexicographic():
    V=np.array([[0,1,1,1],[0,0,0,1],[0,0,1,1]]); np.testing.assert_array_equal(solution.select_smooth_preference(V),[0,0,0,1])
def test_nfl_average_unseen_error_equal_half():
    result=solution.nfl_unseen_error_experiment(4,np.array([0,3])); assert result["n_targets"]==16; assert result["learner_zero"]==pytest.approx(.5); assert result["learner_parity"]==pytest.approx(.5)
def test_different_train_split_keeps_nfl_equality():
    result=solution.nfl_unseen_error_experiment(5,np.array([1,2,4])); assert result["learner_zero"]==pytest.approx(result["learner_parity"])
def test_functions_do_not_modify_inputs():
    H=solution.enumerate_binary_hypotheses(3); H0=H.copy(); idx=np.array([0]); idx0=idx.copy(); solution.version_space(H,idx,np.array([1])); np.testing.assert_array_equal(H,H0); np.testing.assert_array_equal(idx,idx0)
@pytest.mark.parametrize("n",[0,-1,1.5,True])
def test_bad_n_rejected(n):
    with pytest.raises(ValueError): solution.enumerate_binary_hypotheses(n)
def test_bad_observations_rejected():
    H=solution.enumerate_binary_hypotheses(3)
    with pytest.raises(ValueError): solution.version_space(H,np.array([0,0]),np.array([0,1]))
    with pytest.raises(ValueError): solution.version_space(H,np.array([3]),np.array([0]))
    with pytest.raises(ValueError): solution.version_space(H,np.array([0]),np.array([2]))
def test_nfl_requires_nonempty_proper_training_subset():
    with pytest.raises(ValueError): solution.nfl_unseen_error_experiment(3,np.array([],dtype=int))
    with pytest.raises(ValueError): solution.nfl_unseen_error_experiment(3,np.arange(3))
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "hypothesis space shape: (16, 4)" in result.stdout and "version space size: 4" in result.stdout and "'learner_zero': 0.5" in result.stdout
