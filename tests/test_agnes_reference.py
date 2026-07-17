import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TOPIC=ROOT/"watermelon_book"/"09_clustering"/"04_agnes"
spec=importlib.util.spec_from_file_location("agnes_solution",TOPIC/"reference"/"solution.py")
assert spec is not None and spec.loader is not None
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def data(): return np.array([[0.],[1.],[4.],[5.],[10.]])


def test_pairwise_euclidean_shape_symmetry_and_values() -> None:
    distances=solution.pairwise_euclidean(data())
    assert distances.shape==(5,5); np.testing.assert_allclose(distances,distances.T); np.testing.assert_allclose(np.diag(distances),0)
    assert distances[1,4]==pytest.approx(9)


@pytest.mark.parametrize(("linkage","expected"),[("single",3),("complete",5),("average",4)])
def test_cluster_linkages_match_cross_pair_hand_values(linkage,expected) -> None:
    distances=solution.pairwise_euclidean(data())
    assert solution.cluster_distance((0,1),(2,3),distances,linkage)==pytest.approx(expected)


def test_first_tied_merge_uses_earlier_cluster_ids() -> None:
    model=solution.fit_agnes(data(),2,linkage="single")
    first=model["history"][0]
    assert (first["left"],first["right"],first["new"],first["members"])==(0,1,5,(0,1))
    assert first["distance"]==pytest.approx(1)


@pytest.mark.parametrize(("linkage","last_distance"),[("single",3),("complete",5),("average",4)])
def test_three_linkages_record_expected_merge_distances(linkage,last_distance) -> None:
    model=solution.fit_agnes(data(),2,linkage=linkage)
    distances=[step["distance"] for step in model["history"]]
    np.testing.assert_allclose(distances,[1,1,last_distance])
    assert np.all(np.diff(distances)>=-1e-12)


def test_target_two_clusters_members_labels_and_history_length() -> None:
    model=solution.fit_agnes(data(),2,linkage="average")
    assert model["clusters"]==((0,1,2,3),(4,))
    np.testing.assert_array_equal(model["labels"],[0,0,0,0,1])
    assert len(model["history"])==3


def test_target_one_cluster_contains_all_samples_and_n_minus_one_merges() -> None:
    model=solution.fit_agnes(data(),1)
    assert model["clusters"]==((0,1,2,3,4),)
    np.testing.assert_array_equal(model["labels"],np.zeros(5,dtype=int))
    assert len(model["history"])==4


def test_target_n_clusters_has_singletons_and_empty_history() -> None:
    model=solution.fit_agnes(data(),5)
    assert model["clusters"]==((0,),(1,),(2,),(3,),(4,)); assert model["history"]==()
    np.testing.assert_array_equal(model["labels"],np.arange(5))


def test_final_labels_are_ordered_by_smallest_member_not_new_id() -> None:
    X=np.array([[10.],[0.],[1.]])
    model=solution.fit_agnes(X,2)
    assert model["clusters"]==((0,),(1,2)); np.testing.assert_array_equal(model["labels"],[0,1,1])


def test_fit_is_deterministic_and_does_not_modify_input() -> None:
    X=data(); X0=X.copy(); first=solution.fit_agnes(X,2); second=solution.fit_agnes(X,2)
    assert first["history"]==second["history"] and first["clusters"]==second["clusters"]
    np.testing.assert_array_equal(X,X0)


@pytest.mark.parametrize("linkage",["ward","centroid","",None])
def test_invalid_linkage_is_rejected(linkage) -> None:
    with pytest.raises(ValueError): solution.fit_agnes(data(),2,linkage=linkage)


@pytest.mark.parametrize("k",[0,-1,6,1.5,True])
def test_invalid_target_cluster_count_is_rejected(k) -> None:
    with pytest.raises(ValueError): solution.fit_agnes(data(),k)


def test_cluster_distance_rejects_overlap_duplicate_bad_index_and_bad_matrix() -> None:
    distances=solution.pairwise_euclidean(data())
    with pytest.raises(ValueError): solution.cluster_distance((0,1),(1,2),distances,"single")
    with pytest.raises(ValueError): solution.cluster_distance((0,0),(1,),distances,"single")
    with pytest.raises(ValueError): solution.cluster_distance((0,),(9,),distances,"single")
    with pytest.raises(ValueError): solution.cluster_distance((0,),(1,),np.ones((2,3)),"single")


def test_fit_rejects_bad_X() -> None:
    with pytest.raises(ValueError): solution.fit_agnes(np.array([1.,2.]),1)
    with pytest.raises(ValueError): solution.fit_agnes(np.array([[np.nan]]),1)


def test_guided_demo_runs_and_reports_three_linkages() -> None:
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    for name in ("single","complete","average"): assert f"linkage: {name}" in result.stdout
    assert "merge distances: [1.0, 1.0, 3.0]" in result.stdout
    assert "merge distances: [1.0, 1.0, 5.0]" in result.stdout
    assert "merge distances: [1.0, 1.0, 4.0]" in result.stdout

