import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"05_neural_networks"/"09_elman_network"
spec=importlib.util.spec_from_file_location("elman_bptt_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def tiny():
    X=np.array([[.2,-.1],[.0,.3],[-.2,.1]]); targets=np.array([[.1],[-.2],[.3]]); parameters=solution.initialize_elman_parameters(2,2,1,seed=4); return X,targets,parameters
def test_bptt_shapes_loss_and_outputs_match_forward():
    X,y,p=tiny(); result=solution.elman_bptt(X,y,p); assert set(result["gradients"])==solution.PARAMETER_KEYS
    for key in p: assert result["gradients"][key].shape==p[key].shape
    assert result["initial_state_gradient"].shape==(2,); assert result["loss"]==pytest.approx(solution.sequence_mean_squared_error(y,result["outputs"]))
def test_all_parameter_gradients_match_finite_difference():
    X,y,p=tiny(); errors=solution.gradient_check_elman(X,y,p); assert set(errors)==solution.PARAMETER_KEYS; assert max(errors.values())<1e-6
def test_nonzero_initial_state_gradient_and_gradient_check():
    X,y,p=tiny(); initial=np.array([.1,-.2]); result=solution.elman_bptt(X,y,p,initial_state=initial); assert np.linalg.norm(result["initial_state_gradient"])>0; assert max(solution.gradient_check_elman(X,y,p,initial_state=initial).values())<1e-6
def test_delayed_target_produces_recurrent_gradient():
    X,y,p=tiny(); y=np.zeros_like(y); y[-1]=1; assert np.linalg.norm(solution.elman_bptt(X,y,p)["gradients"]["Wh"])>0
def test_training_history_shapes_and_loss_decreases():
    X=np.array([[1.],[0.],[-1.],[.5]]); y=np.array([[.5],[.2],[-.4],[.1]]); model=solution.train_elman_sequence(X,y,3,learning_rate=.05,epochs=400,seed=2); assert model["loss_history"].shape==(401,); assert model["gradient_norms"].shape==(400,); assert model["loss_history"][-1]<model["loss_history"][0]*.2
def test_gradient_clipping_records_unclipped_norm_and_keeps_finite():
    X=np.full((5,1),10.); y=np.full((5,1),100.); model=solution.train_elman_sequence(X,y,2,learning_rate=.1,epochs=10,seed=1,clip_norm=.01); assert np.any(model["gradient_norms"]>.01); assert np.all(np.isfinite(model["loss_history"]))
def test_training_repeatable_and_does_not_modify_inputs():
    X=np.array([[1.],[0.],[-1.]]); y=np.array([[.2],[.1],[-.2]]); X0=X.copy(); a=solution.train_elman_sequence(X,y,2,epochs=10,seed=3); b=solution.train_elman_sequence(X,y,2,epochs=10,seed=3); np.testing.assert_allclose(a["loss_history"],b["loss_history"]); np.testing.assert_array_equal(X,X0)
@pytest.mark.parametrize(("name","value"),[("n_hidden",0),("epochs",0),("learning_rate",0),("clip_norm",0)])
def test_bad_training_options_rejected(name,value):
    X=np.ones((2,1)); y=np.ones((2,1))
    hidden = value if name == "n_hidden" else 2
    options = {} if name == "n_hidden" else {name: value}
    with pytest.raises(ValueError): solution.train_elman_sequence(X,y,hidden,**options)
def test_bad_targets_and_epsilon_rejected():
    X,y,p=tiny()
    with pytest.raises(ValueError): solution.elman_bptt(X,y.ravel(),p)
    with pytest.raises(ValueError): solution.gradient_check_elman(X,y,p,epsilon=0)
