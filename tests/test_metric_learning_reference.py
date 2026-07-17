import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"10_dimensionality_reduction"/"06_metric_learning"
spec=importlib.util.spec_from_file_location("metric_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def data(): return np.array([[0.,0.],[0.,2.],[3.,0.],[3.,2.]]),np.array([0,0,1,1])
def test_identity_matches_euclidean():
    X,_=data(); D=solution.pairwise_mahalanobis(X,np.eye(2)); assert D[0,3]==pytest.approx(np.sqrt(13)); np.testing.assert_allclose(D,D.T); np.testing.assert_allclose(np.diag(D),0)
def test_diagonal_metric_scales_features():
    X=np.array([[0.,0.],[2.,3.]]); assert solution.pairwise_mahalanobis(X,np.diag([4.,0.]))[0,1]==pytest.approx(4)
def test_psd_projection_clips_negative_eigenvalues():
    M=solution.project_psd(np.array([[1.,2.],[2.,1.]])); assert np.min(np.linalg.eigvalsh(M))>=-1e-12; np.testing.assert_allclose(M,M.T)
def test_fit_objective_nonincreasing_and_psd():
    X,y=data(); model=solution.fit_pair_metric(X,y,margin=20,regularization=.1)
    assert np.all(np.diff(model["objective_history"])<=1e-10); assert model["objective_history"][-1]<model["objective_history"][0]; assert np.min(np.linalg.eigvalsh(model["metric"]))>=-1e-10
def test_learning_emphasizes_between_class_x_direction():
    X,y=data(); M=solution.fit_pair_metric(X,y,margin=20,regularization=.1)["metric"]; assert M[0,0]>M[1,1]
def test_learned_metric_changes_with_margin():
    X,y=data(); small=solution.fit_pair_metric(X,y,margin=1,regularization=.1)["metric"]; large=solution.fit_pair_metric(X,y,margin=20,regularization=.1)["metric"]; assert not np.allclose(small,large)
def test_repeatable_and_input_not_modified():
    X,y=data(); X0=X.copy(); a=solution.fit_pair_metric(X,y,margin=20); b=solution.fit_pair_metric(X,y,margin=20); np.testing.assert_allclose(a["metric"],b["metric"]); np.testing.assert_array_equal(X,X0)
def test_objective_public_matches_history_start():
    X,y=data(); model=solution.fit_pair_metric(X,y,margin=20,regularization=.1); assert model["objective_history"][0]==pytest.approx(solution.metric_objective(X,y,np.eye(2),margin=20,regularization=.1))
def test_non_psd_metric_rejected():
    X,_=data()
    with pytest.raises(ValueError): solution.pairwise_mahalanobis(X,np.diag([1.,-1.]))
def test_bad_labels_and_pairs_rejected():
    X,_=data()
    with pytest.raises(ValueError): solution.fit_pair_metric(X,np.zeros(4,dtype=int))
    with pytest.raises(ValueError): solution.fit_pair_metric(np.array([[0.],[1.]]),np.array([0,1]))
@pytest.mark.parametrize(("name","value"),[("margin",0),("different_weight",-1),("regularization",-1),("learning_rate",0),("tolerance",0),("max_iterations",0)])
def test_bad_parameters_rejected(name,value):
    X,y=data(); options={name:value}
    with pytest.raises(ValueError): solution.fit_pair_metric(X,y,**options)
def test_bad_X_and_metric_shape_rejected():
    X,y=data()
    with pytest.raises(ValueError): solution.fit_pair_metric(np.array([1.,2.]),np.array([0,1]))
    with pytest.raises(ValueError): solution.pairwise_mahalanobis(X,np.eye(3))
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "metric shape: (2, 2)" in result.stdout and "x weight > y weight: True" in result.stdout
