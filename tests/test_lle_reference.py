import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"10_dimensionality_reduction"/"07_lle"
spec=importlib.util.spec_from_file_location("lle_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def curve():
    t=np.linspace(-1.2,1.2,9); return np.column_stack((t,t*t))
def test_neighbor_shape_excludes_self_and_stable_tie():
    X=np.array([[0.],[1.],[-1.],[3.]]); neighbors=solution.nearest_neighbors(X,1); assert neighbors.shape==(4,1); assert neighbors[0,0]==1; assert np.all(neighbors[:,0]!=np.arange(4))
def test_weights_rows_sum_non_neighbors_zero():
    X=curve(); neighbors=solution.nearest_neighbors(X,2); W=solution.reconstruction_weights(X,neighbors); np.testing.assert_allclose(W.sum(axis=1),1)
    for i,row in enumerate(neighbors): assert np.count_nonzero(W[i])<=2 and all(W[i,j]==0 for j in range(len(X)) if j not in row)
def test_two_neighbor_line_reconstructs_interior():
    X=np.arange(5.).reshape(-1,1); neighbors=np.array([[1,2],[0,2],[1,3],[2,4],[3,2]]); W=solution.reconstruction_weights(X,neighbors,regularization=1e-8); np.testing.assert_allclose((W@X)[1:4],X[1:4],atol=1e-6)
def test_fit_shapes_center_scale_and_trivial_eigenvalue():
    model=solution.fit_lle(curve(),2,1); assert model["coordinates"].shape==(9,1); assert model["weights"].shape==(9,9); np.testing.assert_allclose(model["coordinates"].mean(axis=0),0,atol=1e-12); np.testing.assert_allclose(np.mean(model["coordinates"]**2,axis=0),1,atol=1e-12); assert abs(model["eigenvalues"][0])<1e-10
def test_embedding_matrix_psd_and_low_error_finite():
    model=solution.fit_lle(curve(),2,1); assert np.min(np.linalg.eigvalsh(model["embedding_matrix"]))>=-1e-10; assert np.isfinite(model["low_dimensional_error"])
def test_translation_and_rotation_preserve_neighbors_and_weights():
    X=curve(); rotation=np.array([[0.,-1.],[1.,0.]]); transformed=X@rotation+np.array([5.,-3.]); a=solution.fit_lle(X,2,1); b=solution.fit_lle(transformed,2,1); np.testing.assert_array_equal(a["neighbors"],b["neighbors"]); np.testing.assert_allclose(a["weights"],b["weights"],atol=1e-10)
def test_repeatable_and_input_not_modified():
    X=curve(); X0=X.copy(); a=solution.fit_lle(X,2,1); b=solution.fit_lle(X,2,1); np.testing.assert_allclose(a["coordinates"],b["coordinates"]); np.testing.assert_array_equal(X,X0)
def test_regularization_handles_more_neighbors_than_dimensions():
    model=solution.fit_lle(curve(),4,1); assert np.all(np.isfinite(model["weights"]))
@pytest.mark.parametrize("k",[0,-1,9,1.5,True])
def test_bad_neighbors_rejected(k):
    with pytest.raises(ValueError): solution.nearest_neighbors(curve(),k)
@pytest.mark.parametrize("q",[0,-1,9,1.5,True])
def test_bad_components_rejected(q):
    with pytest.raises(ValueError): solution.fit_lle(curve(),2,q)
def test_bad_neighbor_matrix_and_regularization_rejected():
    X=curve()
    with pytest.raises(ValueError): solution.reconstruction_weights(X,np.zeros((9,2),dtype=int))
    with pytest.raises(ValueError): solution.fit_lle(X,2,1,regularization=0)
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "neighbor shape: (9, 2)" in result.stdout and "coordinates shape: (9, 1)" in result.stdout and "coordinate mean: [0.0]" in result.stdout
