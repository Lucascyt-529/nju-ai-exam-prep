import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"05_neural_networks"/"10_cascade_correlation"
spec=importlib.util.spec_from_file_location("cc_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def data():
    X=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.],[0.,-1.],[0.,1.]])
    return X,X[:,0]*X[:,1]
def test_absolute_correlation_hand_perfect_and_zero():
    residual=np.array([-1.,0.,1.]); A=np.column_stack((residual,np.ones(3),-residual)); np.testing.assert_allclose(solution.absolute_correlation(A,residual),[1,0,1])
def test_constant_residual_returns_zero():
    assert np.all(solution.absolute_correlation(np.arange(6.).reshape(3,2),np.ones(3))==0)
def test_hidden_width_grows_and_output_matches():
    X,y=data(); model=solution.fit_cascade_correlation(X,y,n_hidden=3,n_candidates=50)
    assert [len(u["weights"]) for u in model["hidden_units"]]==[2,3,4]; assert len(model["output_weights"])==5
def test_mse_history_length_nonincreasing_and_improves_nonlinear_fit():
    X,y=data(); model=solution.fit_cascade_correlation(X,y,n_hidden=3,n_candidates=100,random_state=7)
    assert len(model["mse_history"])==4; assert np.all(np.diff(model["mse_history"])<=1e-9); assert model["mse_history"][-1]<model["mse_history"][0]*.2
def test_prediction_reproduces_recorded_final_mse():
    X,y=data(); model=solution.fit_cascade_correlation(X,y,n_hidden=2,random_state=4); pred=solution.predict_cascade(model,X); assert pred.shape==(6,); assert np.mean((y-pred)**2)==pytest.approx(model["mse_history"][-1])
def test_zero_hidden_is_linear_baseline():
    X,y=data(); model=solution.fit_cascade_correlation(X,y,n_hidden=0); assert model["hidden_units"]==(); assert len(model["mse_history"])==1; assert len(model["output_weights"])==2
def test_repeatable_and_input_not_modified():
    X,y=data(); X0=X.copy(); a=solution.fit_cascade_correlation(X,y,random_state=3); b=solution.fit_cascade_correlation(X,y,random_state=3); np.testing.assert_allclose(a["output_weights"],b["output_weights"]); np.testing.assert_allclose(a["mse_history"],b["mse_history"]); np.testing.assert_array_equal(X,X0)
def test_selected_correlations_valid():
    X,y=data(); scores=solution.fit_cascade_correlation(X,y)["selected_correlations"]; assert scores.shape==(3,) and np.all((scores>=0)&(scores<=1+1e-12))
@pytest.mark.parametrize(("name","value"),[("n_hidden",-1),("n_candidates",0),("ridge",-1),("random_state",1.5)])
def test_bad_options_rejected(name,value):
    X,y=data()
    with pytest.raises(ValueError): solution.fit_cascade_correlation(X,y,**{name:value})
def test_bad_X_y_and_model_rejected():
    X,y=data()
    with pytest.raises(ValueError): solution.fit_cascade_correlation(X,np.ones((6,1)))
    model=solution.fit_cascade_correlation(X,y,n_hidden=1)
    with pytest.raises(ValueError): solution.predict_cascade({},X)
    with pytest.raises(ValueError): solution.predict_cascade(model,np.ones((2,3)))
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "hidden input widths: [2, 3, 4]" in result.stdout and "output width: 5" in result.stdout and "prediction shape: (6,)" in result.stdout
