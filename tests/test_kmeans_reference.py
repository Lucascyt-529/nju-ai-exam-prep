import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TOPIC=ROOT/"02_machine_learning"/"05_kmeans"
spec=importlib.util.spec_from_file_location("kmeans_solution",TOPIC/"reference"/"solution.py")
assert spec is not None and spec.loader is not None
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def data(): return np.array([[0.,0.],[0.,1.],[1.,0.],[8.,8.],[8.,9.],[9.,8.]])


def test_squared_distance_matrix_matches_hand_values_and_shape() -> None:
    X=np.array([[0.,0.],[1.,2.]]); centers=np.array([[0.,1.],[3.,4.]])
    np.testing.assert_allclose(solution.squared_distance_matrix(X,centers),[[1,25],[2,8]])
    assert solution.squared_distance_matrix(X,centers).shape==(2,2)


def test_assignment_uses_nearest_center_and_earliest_tie() -> None:
    X=np.array([[0.],[1.],[2.]]); centers=np.array([[0.],[2.]])
    labels,distances=solution.assign_labels(X,centers)
    np.testing.assert_array_equal(labels,[0,0,1]); assert distances.shape==(3,2)


def test_kmeans_plus_plus_is_reproducible_unique_and_from_data() -> None:
    X=data(); first=solution.kmeans_plus_plus(X,2,random_state=4); second=solution.kmeans_plus_plus(X,2,random_state=4)
    np.testing.assert_array_equal(first,second); assert np.unique(first,axis=0).shape[0]==2
    assert all(np.any(np.all(X==center,axis=1)) for center in first)
    np.testing.assert_array_equal(first,[[8.,9.],[0.,1.]])


def test_duplicate_samples_do_not_duplicate_selected_coordinates() -> None:
    X=np.array([[0.],[0.],[5.],[5.],[10.]])
    centers=solution.kmeans_plus_plus(X,3,random_state=2)
    assert np.unique(centers,axis=0).shape[0]==3


def test_center_update_is_cluster_mean() -> None:
    X=np.array([[0.],[2.],[10.],[14.]]); labels=np.array([0,0,1,1]); old=np.array([[0.],[10.]])
    np.testing.assert_allclose(solution.update_centers(X,labels,old),[[1.],[12.]])


def test_empty_cluster_resets_to_largest_current_error_with_early_tie() -> None:
    X=np.array([[0.],[2.],[10.]]); labels=np.array([0,0,1]); old=np.array([[0.],[10.],[100.]])
    updated=solution.update_centers(X,labels,old)
    # 样本2到旧中心0距离最大(4)，空簇2重置到该样本。
    np.testing.assert_allclose(updated,[[1.],[10.],[2.]])


def test_inertia_matches_assigned_squared_distance_sum() -> None:
    X=np.array([[0.],[2.],[10.]]); centers=np.array([[1.],[10.]]); labels=np.array([0,0,1])
    assert solution.inertia(X,centers,labels)==pytest.approx(2.0)


def test_fit_finds_two_well_separated_clusters_and_centers() -> None:
    X=data(); model=solution.fit_kmeans(X,2,random_state=4)
    np.testing.assert_array_equal(model["labels"],[1,1,1,0,0,0])
    np.testing.assert_allclose(model["centers"],[[25/3,25/3],[1/3,1/3]])
    assert model["iterations"]>=1


def test_inertia_history_is_nondecreasing_false_and_matches_final() -> None:
    X=data(); model=solution.fit_kmeans(X,2,random_state=4)
    assert np.all(np.diff(model["inertia_history"])<=1e-12)
    assert model["inertia_history"][-1]==pytest.approx(solution.inertia(X,model["centers"],model["labels"]))


def test_one_cluster_returns_global_mean() -> None:
    X=data(); model=solution.fit_kmeans(X,1,random_state=1)
    np.testing.assert_allclose(model["centers"],[X.mean(axis=0)])
    np.testing.assert_array_equal(model["labels"],np.zeros(len(X),dtype=int))


def test_prediction_uses_fitted_centers_and_shape() -> None:
    X=data(); model=solution.fit_kmeans(X,2,random_state=4)
    pred=solution.predict(model,np.array([[0.,0.],[9.,9.]]))
    np.testing.assert_array_equal(pred,[1,0]); assert pred.shape==(2,)


def test_fit_is_deterministic_and_does_not_modify_input() -> None:
    X=data(); X0=X.copy(); first=solution.fit_kmeans(X,2,random_state=7); second=solution.fit_kmeans(X,2,random_state=7)
    np.testing.assert_array_equal(first["centers"],second["centers"]); np.testing.assert_array_equal(first["labels"],second["labels"])
    np.testing.assert_array_equal(X,X0)


@pytest.mark.parametrize("k",[0,-1,1.5,True,7])
def test_kmeans_rejects_invalid_cluster_count(k) -> None:
    with pytest.raises(ValueError): solution.fit_kmeans(data(),k)


def test_kmeans_plus_plus_rejects_more_clusters_than_unique_points() -> None:
    with pytest.raises(ValueError,match="不同坐标"):
        solution.kmeans_plus_plus(np.array([[0.],[0.],[1.]]),3)


@pytest.mark.parametrize("tolerance",[-1,np.inf,True])
def test_fit_rejects_invalid_tolerance(tolerance) -> None:
    with pytest.raises(ValueError): solution.fit_kmeans(data(),2,tolerance=tolerance)


def test_functions_reject_bad_shapes_feature_mismatch_and_labels() -> None:
    with pytest.raises(ValueError): solution.fit_kmeans(np.array([1.,2.]),2)
    with pytest.raises(ValueError): solution.squared_distance_matrix(np.ones((2,2)),np.ones((2,1)))
    with pytest.raises(ValueError): solution.update_centers(np.ones((2,1)),np.array([0,2]),np.ones((2,1)))
    model=solution.fit_kmeans(data(),2)
    with pytest.raises(ValueError): solution.predict(model,np.ones((2,3)))


def test_guided_demo_runs_and_reports_nonincreasing_history() -> None:
    result=subprocess.run([sys.executable,str(TOPIC/"reference_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "initial centers: [[8.0, 9.0], [0.0, 1.0]]" in result.stdout
    assert "distance shape: (6, 2)" in result.stdout
    assert "labels: [1, 1, 1, 0, 0, 0]" in result.stdout
    assert "nonincreasing: True" in result.stdout
