import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"10_dimensionality_reduction"/"08_nca"
spec=importlib.util.spec_from_file_location("nca_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def data(): return np.array([[-1.,-2.],[-.8,2.],[-.6,0.],[.6,-2.],[.8,2.],[1.,0.]]),np.array([0,0,0,1,1,1])
def test_probability_shape_rows_diagonal_and_asymmetry():
    X=np.array([[0.],[1.],[3.]]); P=solution.neighbor_probabilities(X,np.eye(1)); assert P.shape==(3,3); np.testing.assert_allclose(P.sum(axis=1),1); np.testing.assert_allclose(np.diag(P),0); assert not np.allclose(P,P.T)
def test_zero_transformation_is_uniform_over_other_samples():
    X,y=data(); P=solution.neighbor_probabilities(X,np.zeros((2,2))); expected=(np.ones((6,6))-np.eye(6))/5; np.testing.assert_allclose(P,expected)
def test_objective_matches_mean_correct_probability():
    X,y=data(); A=np.eye(2); P=solution.neighbor_probabilities(X,A); manual=np.mean([P[i,y==y[i]].sum() for i in range(len(y))]); assert solution.nca_objective(X,y,A)==pytest.approx(manual)
def test_analytic_gradient_matches_finite_difference():
    X,y=data(); A=np.array([[.8,.1],[-.2,.6]]); analytic=solution.nca_gradient(X,y,A); numeric=np.zeros_like(A); epsilon=1e-6
    for index in np.ndindex(A.shape):
        plus=A.copy(); minus=A.copy(); plus[index]+=epsilon; minus[index]-=epsilon; numeric[index]=(solution.nca_objective(X,y,plus)-solution.nca_objective(X,y,minus))/(2*epsilon)
    np.testing.assert_allclose(analytic,numeric,atol=1e-6,rtol=1e-5)
def test_fit_objective_nondecreasing_and_improves():
    X,y=data(); model=solution.fit_nca(X,y,learning_rate=.2,max_iterations=200); assert np.all(np.diff(model["objective_history"])>=-1e-10); assert model["objective_history"][-1]>model["objective_history"][0]+.1
def test_metric_psd_and_noise_weight_reduced():
    X,y=data(); M=solution.fit_nca(X,y,learning_rate=.2,max_iterations=200)["metric"]; assert np.min(np.linalg.eigvalsh(M))>=-1e-10; assert M[0,0]>M[1,1]
def test_returned_probabilities_and_correct_values_consistent():
    X,y=data(); model=solution.fit_nca(X,y,max_iterations=10); P=solution.neighbor_probabilities(X,model["transformation"]); np.testing.assert_allclose(P,model["probabilities"]); assert model["correct_probabilities"].mean()==pytest.approx(model["objective_history"][-1])
def test_repeatable_and_input_not_modified():
    X,y=data(); X0=X.copy(); a=solution.fit_nca(X,y,max_iterations=20); b=solution.fit_nca(X,y,max_iterations=20); np.testing.assert_allclose(a["transformation"],b["transformation"]); np.testing.assert_array_equal(X,X0)
def test_translation_invariant_objective_and_gradient():
    X,y=data(); A=np.eye(2); shifted=X+np.array([5.,-3.]); assert solution.nca_objective(X,y,A)==pytest.approx(solution.nca_objective(shifted,y,A)); np.testing.assert_allclose(solution.nca_gradient(X,y,A),solution.nca_gradient(shifted,y,A))
def test_bad_X_y_and_transformation_rejected():
    X,y=data()
    with pytest.raises(ValueError): solution.nca_objective(X,np.zeros(6,dtype=int),np.eye(2))
    with pytest.raises(ValueError): solution.neighbor_probabilities(X,np.eye(3))
@pytest.mark.parametrize(("name","value"),[("learning_rate",0),("max_iterations",0),("tolerance",0)])
def test_bad_fit_options_rejected(name,value):
    X,y=data()
    with pytest.raises(ValueError): solution.fit_nca(X,y,**{name:value})
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "probability shape: (6, 6)" in result.stdout and "metric minimum eigenvalue:" in result.stdout
