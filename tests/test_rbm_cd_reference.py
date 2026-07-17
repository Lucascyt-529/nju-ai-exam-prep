import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"05_neural_networks"/"13_rbm_cd"
spec=importlib.util.spec_from_file_location("rbm_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def data(): return np.array([[1,1,0,0],[1,0,0,0],[0,0,1,1],[0,0,0,1]])
def params(): return np.array([[.2,-.1],[.3,.4],[-.2,.1],[.1,-.3]]),np.zeros(4),np.zeros(2)
def test_hidden_and_visible_probability_shapes_ranges():
    V=data(); W,vb,hb=params(); hp=solution.hidden_probabilities(V,W,hb); assert hp.shape==(4,2) and np.all((hp>0)&(hp<1)); vp=solution.visible_probabilities(np.array([[1,0],[0,1]]),W,vb); assert vp.shape==(2,4) and np.all((vp>0)&(vp<1))
def test_zero_parameters_give_half_probabilities():
    V=data(); np.testing.assert_allclose(solution.hidden_probabilities(V,np.zeros((4,2)),np.zeros(2)),.5); np.testing.assert_allclose(solution.visible_probabilities(np.zeros((3,2),dtype=int),np.zeros((4,2)),np.zeros(4)),.5)
def test_cd1_state_shapes_binary_and_probabilities():
    V=data(); W,vb,hb=params(); step=solution.cd1_step(V,W,vb,hb,random_state=3)
    assert step["hidden_state_0"].shape==(4,2) and step["visible_state_1"].shape==(4,4) and step["hidden_state_1"].shape==(4,2); assert np.all(np.isin(step["visible_state_1"],[0,1]))
def test_cd1_gradients_match_returned_sample_statistics():
    V=data(); W,vb,hb=params(); step=solution.cd1_step(V,W,vb,hb,learning_rate=.2,random_state=3); n=len(V); expected=V.T@step["hidden_state_0"]/n-step["visible_state_1"].T@step["hidden_state_1"]/n; np.testing.assert_allclose(step["gradients"]["weights"],expected); np.testing.assert_allclose(step["weights"],W+.2*expected)
def test_bias_gradients_match_sample_means():
    V=data(); W,vb,hb=params(); step=solution.cd1_step(V,W,vb,hb,random_state=2); np.testing.assert_allclose(step["gradients"]["visible_bias"],np.mean(V-step["visible_state_1"],axis=0)); np.testing.assert_allclose(step["gradients"]["hidden_bias"],np.mean(step["hidden_state_0"]-step["hidden_state_1"],axis=0))
def test_cd1_repeatable_and_inputs_not_modified():
    V=data(); W,vb,hb=params(); copies=[x.copy() for x in (V,W,vb,hb)]; a=solution.cd1_step(V,W,vb,hb,random_state=5); b=solution.cd1_step(V,W,vb,hb,random_state=5); np.testing.assert_allclose(a["weights"],b["weights"])
    for value,copy in zip((V,W,vb,hb),copies): np.testing.assert_array_equal(value,copy)
def test_fit_shapes_history_finite_and_repeatable():
    V=data(); a=solution.fit_rbm_cd1(V,2,epochs=20,random_state=4); b=solution.fit_rbm_cd1(V,2,epochs=20,random_state=4); assert a["weights"].shape==(4,2); assert a["visible_bias"].shape==(4,) and a["hidden_bias"].shape==(2,); assert a["reconstruction_history"].shape==(21,) and np.all(np.isfinite(a["reconstruction_history"])); np.testing.assert_allclose(a["weights"],b["weights"])
def test_fit_usually_improves_tiny_pattern_reconstruction_without_monotonic_claim():
    model=solution.fit_rbm_cd1(data(),2,epochs=200,learning_rate=.1,random_state=3); assert model["reconstruction_history"][-1]<model["reconstruction_history"][0]
def test_bad_binary_data_and_parameter_shapes_rejected():
    W,vb,hb=params()
    with pytest.raises(ValueError): solution.hidden_probabilities(np.array([[.5,0,0,1]]),W,hb)
    with pytest.raises(ValueError): solution.cd1_step(data(),np.ones((3,2)),vb,hb)
@pytest.mark.parametrize(("name","value"),[("n_hidden",0),("epochs",0),("learning_rate",0),("random_state",1.5)])
def test_bad_fit_options_rejected(name,value):
    arguments={"n_hidden":2,"epochs":100,"learning_rate":.1,"random_state":0}
    arguments[name]=value
    with pytest.raises(ValueError): solution.fit_rbm_cd1(data(),**arguments)
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "weight shape: (4, 2)" in result.stdout and "hidden probability shape: (4, 2)" in result.stdout and "parameters finite: True" in result.stdout
