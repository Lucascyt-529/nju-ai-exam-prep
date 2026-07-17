import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"09_clustering"/"06_gaussian_mixture"
spec=importlib.util.spec_from_file_location("gmm_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def data(): return np.array([[-3.,0.],[-2.8,.2],[-3.2,-.1],[3.,0.],[2.8,-.2],[3.2,.1]])
def test_log_density_shape_and_standard_normal_value():
    value=solution.gaussian_log_density(np.array([[0.,0.]]),np.array([[0.,0.]]),np.array([np.eye(2)])); assert value.shape==(1,1); assert value[0,0]==pytest.approx(-np.log(2*np.pi))
def test_e_step_shape_rows_and_likelihood():
    X=data(); w=np.array([.5,.5]); m=np.array([[-3.,0.],[3.,0.]]); c=np.array([np.eye(2),np.eye(2)]); r,ll=solution.expectation_step(X,w,m,c); assert r.shape==(6,2); np.testing.assert_allclose(r.sum(axis=1),1); assert np.isfinite(ll); assert r[0,0]>.99 and r[-1,1]>.99
def test_m_step_soft_counts_means_covariance_shapes():
    X=data(); r=np.zeros((6,2)); r[:3,0]=1; r[3:,1]=1; w,m,c=solution.maximization_step(X,r); np.testing.assert_allclose(w,[.5,.5]); np.testing.assert_allclose(m[:,0],[-3,3]); assert c.shape==(2,2,2); assert np.all(np.linalg.eigvalsh(c)>0)
def test_fit_likelihood_nondecreasing_and_separates_clusters():
    model=solution.fit_gaussian_mixture(data(),2,random_state=4); assert np.all(np.diff(model["log_likelihood_history"])>=-1e-7); np.testing.assert_allclose(model["weights"].sum(),1); assert np.all(model["labels"][:3]==0) and np.all(model["labels"][3:]==1); np.testing.assert_allclose(model["means"][:,0],[-3,3],atol=.2)
def test_final_responsibilities_match_parameters():
    X=data(); model=solution.fit_gaussian_mixture(X,2,random_state=4); r,_=solution.expectation_step(X,model["weights"],model["means"],model["covariances"]); np.testing.assert_allclose(r,model["responsibilities"])
def test_single_component_mean_and_labels():
    X=data(); model=solution.fit_gaussian_mixture(X,1); np.testing.assert_allclose(model["means"][0],X.mean(axis=0)); np.testing.assert_array_equal(model["labels"],np.zeros(6,dtype=int))
def test_repeatable_and_input_not_modified():
    X=data(); X0=X.copy(); a=solution.fit_gaussian_mixture(X,2,random_state=3); b=solution.fit_gaussian_mixture(X,2,random_state=3); np.testing.assert_allclose(a["means"],b["means"]); np.testing.assert_array_equal(X,X0)
def test_covariances_symmetric_positive_definite():
    model=solution.fit_gaussian_mixture(data(),2); np.testing.assert_allclose(model["covariances"],model["covariances"].transpose(0,2,1)); assert np.all(np.linalg.eigvalsh(model["covariances"])>0)
@pytest.mark.parametrize("k",[0,-1,7,1.5,True])
def test_bad_component_count_rejected(k):
    with pytest.raises(ValueError): solution.fit_gaussian_mixture(data(),k)
def test_bad_covariance_and_responsibilities_rejected():
    X=data()
    with pytest.raises(ValueError): solution.gaussian_log_density(X,np.zeros((1,2)),np.array([np.zeros((2,2))]))
    with pytest.raises(ValueError): solution.maximization_step(X,np.ones((6,2)))
@pytest.mark.parametrize(("name","value"),[("max_iterations",0),("tolerance",0),("covariance_regularization",0),("random_state",1.5)])
def test_bad_fit_options_rejected(name,value):
    with pytest.raises(ValueError): solution.fit_gaussian_mixture(data(),2,**{name:value})
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "responsibility shape: (6, 2)" in result.stdout and "labels: [0, 0, 0, 1, 1, 1]" in result.stdout and "likelihood nondecreasing: True" in result.stdout
