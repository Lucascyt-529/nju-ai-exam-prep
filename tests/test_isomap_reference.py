import importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"watermelon_book"/"10_dimensionality_reduction"/"05_isomap"
spec=importlib.util.spec_from_file_location("isomap_solution",TOPIC/"reference"/"solution.py"); assert spec and spec.loader
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
def arc():
    angles=np.linspace(0,np.pi,7); return np.column_stack((np.cos(angles),np.sin(angles)))
def test_pairwise_shape_and_values():
    D=solution.pairwise_euclidean(np.array([[0.,0.],[3.,4.]])); np.testing.assert_allclose(D,[[0,5],[5,0]])
def test_knn_graph_is_symmetric_with_zero_diagonal_and_inf_nonedge():
    G=solution.knn_graph(np.arange(4.).reshape(-1,1),1); np.testing.assert_allclose(G,G.T); np.testing.assert_allclose(np.diag(G),0); assert np.isinf(G[0,2])
def test_tied_neighbor_uses_smaller_index_before_union():
    G=solution.knn_graph(np.array([[0.],[1.],[2.],[2.1]]),1)
    assert np.isfinite(G[1,0]); assert not np.isfinite(G[1,2])
def test_floyd_known_path():
    inf=np.inf; G=np.array([[0,1,inf,inf],[1,0,2,inf],[inf,2,0,3],[inf,inf,3,0]],float)
    D=solution.floyd_warshall(G); assert D[0,3]==pytest.approx(6); np.testing.assert_allclose(D,D.T)
def test_arc_geodesic_exceeds_endpoint_chord():
    model=solution.fit_isomap(arc(),2,1); assert model["geodesic_distances"][0,-1]>model["euclidean_distances"][0,-1]
def test_embedding_shape_centering_and_repeatability():
    a=solution.fit_isomap(arc(),2,1); b=solution.fit_isomap(arc(),2,1)
    assert a["coordinates"].shape==(7,1); np.testing.assert_allclose(a["coordinates"].mean(axis=0),0,atol=1e-12); np.testing.assert_allclose(a["coordinates"],b["coordinates"])
def test_complete_graph_geodesic_equals_euclidean():
    X=arc(); model=solution.fit_isomap(X,len(X)-1,2); np.testing.assert_allclose(model["geodesic_distances"],model["euclidean_distances"])
def test_disconnected_graph_rejected():
    X=np.array([[0.],[1.],[10.],[11.]])
    with pytest.raises(ValueError,match="不连通"): solution.fit_isomap(X,1,1)
def test_input_not_modified():
    X=arc(); original=X.copy(); solution.fit_isomap(X,2,1); np.testing.assert_array_equal(X,original)
@pytest.mark.parametrize("k",[0,-1,7,1.5,True])
def test_bad_neighbors_rejected(k):
    with pytest.raises(ValueError): solution.knn_graph(arc(),k)
@pytest.mark.parametrize("q",[0,-1,8,1.5,True])
def test_bad_components_rejected(q):
    with pytest.raises(ValueError): solution.fit_isomap(arc(),2,q)
def test_bad_graph_rejected():
    with pytest.raises(ValueError): solution.floyd_warshall(np.ones((2,3)))
    with pytest.raises(ValueError): solution.floyd_warshall(np.array([[0,1],[2,0]],float))
def test_guided_demo_runs():
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "graph shape: (7, 7)" in result.stdout and "coordinates shape: (7, 1)" in result.stdout and "coordinates centered: True" in result.stdout
