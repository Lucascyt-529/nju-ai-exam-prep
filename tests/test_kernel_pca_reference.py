import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"10_dimensionality_reduction"/"04_kernel_pca"
spec=importlib.util.spec_from_file_location("kpca_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)

def data(): return np.array([[-2.,-1.],[-2.,1.],[2.,-1.],[2.,1.]])

def test_three_kernel_hand_values():
    X=np.array([[1.,0.],[0.,1.]])
    np.testing.assert_allclose(solution.kernel_matrix(X,X,kernel="linear"),np.eye(2))
    np.testing.assert_allclose(solution.kernel_matrix(X,X,kernel="polynomial",degree=2,gamma=1,coef0=1),[[4,1],[1,4]])
    expected=np.array([[1,np.exp(-2)],[np.exp(-2),1]])
    np.testing.assert_allclose(solution.kernel_matrix(X,X,kernel="rbf",gamma=1),expected)

def test_kernel_shapes_and_cross_features():
    assert solution.kernel_matrix(np.ones((3,2)),np.ones((4,2))).shape==(3,4)
    with pytest.raises(ValueError): solution.kernel_matrix(np.ones((3,2)),np.ones((4,3)))

def test_center_kernel_zero_means_and_saved_statistics():
    K=solution.kernel_matrix(data(),data(),kernel="linear"); centered,means,grand=solution.center_train_kernel(K)
    np.testing.assert_allclose(centered.mean(axis=0),0,atol=1e-12); np.testing.assert_allclose(centered.mean(axis=1),0,atol=1e-12)
    np.testing.assert_allclose(means,K.mean(axis=0)); assert grand==pytest.approx(K.mean())

def test_linear_kernel_pca_reconstructs_centered_gram():
    X=data(); model=solution.fit_kernel_pca(X,2,kernel="linear")
    centered=X-X.mean(axis=0); np.testing.assert_allclose(model["coordinates"]@model["coordinates"].T,centered@centered.T,atol=1e-10)

def test_training_reprojection_matches_coordinates():
    X=data(); model=solution.fit_kernel_pca(X,2,kernel="rbf",gamma=.5)
    np.testing.assert_allclose(solution.transform_kernel_pca(model,X),model["coordinates"],atol=1e-10)

def test_new_sample_projection_shape_and_finite():
    model=solution.fit_kernel_pca(data(),2,kernel="rbf",gamma=.5); Z=solution.transform_kernel_pca(model,np.array([[0.,0.],[3.,3.]]))
    assert Z.shape==(2,2); assert np.all(np.isfinite(Z))

def test_eigenvalues_positive_descending_and_coordinates_centered():
    model=solution.fit_kernel_pca(data(),2,kernel="rbf",gamma=.5)
    assert np.all(model["eigenvalues"]>0) and np.all(np.diff(model["eigenvalues"])<=0)
    np.testing.assert_allclose(model["coordinates"].mean(axis=0),0,atol=1e-12)

def test_repeatability_and_input_not_modified():
    X=data(); original=X.copy(); a=solution.fit_kernel_pca(X,2,kernel="rbf"); b=solution.fit_kernel_pca(X,2,kernel="rbf")
    np.testing.assert_allclose(a["coordinates"],b["coordinates"]); np.testing.assert_array_equal(X,original)

def test_insufficient_positive_spectrum_rejected():
    X=np.array([[0.],[1.],[2.]])
    with pytest.raises(ValueError): solution.fit_kernel_pca(X,2,kernel="linear")

@pytest.mark.parametrize("kernel",["sigmoid","",None])
def test_bad_kernel_rejected(kernel):
    with pytest.raises(ValueError): solution.fit_kernel_pca(data(),1,kernel=kernel)

@pytest.mark.parametrize("n_components",[0,-1,5,1.5,True])
def test_bad_component_count_rejected(n_components):
    with pytest.raises(ValueError): solution.fit_kernel_pca(data(),n_components)

def test_bad_kernel_parameters_rejected():
    with pytest.raises(ValueError): solution.fit_kernel_pca(data(),1,kernel="rbf",gamma=0)
    with pytest.raises(ValueError): solution.fit_kernel_pca(data(),1,kernel="polynomial",degree=0)
    with pytest.raises(ValueError): solution.fit_kernel_pca(data(),1,coef0=np.inf)

def test_bad_matrices_and_bad_model_rejected():
    with pytest.raises(ValueError): solution.fit_kernel_pca(np.array([1.,2.]),1)
    model=solution.fit_kernel_pca(data(),1)
    with pytest.raises(ValueError): solution.transform_kernel_pca({},data())
    with pytest.raises(ValueError): solution.transform_kernel_pca(model,np.ones((2,3)))

def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "kernel shape: (8, 8)" in result.stdout
    assert "coordinates shape: (8, 2)" in result.stdout
    assert "training reprojection matches: True" in result.stdout
