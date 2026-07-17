import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TOPIC=ROOT/"watermelon_book"/"09_clustering"/"01_distances_metrics"
spec=importlib.util.spec_from_file_location("clustering_metrics_solution",TOPIC/"reference"/"solution.py")
assert spec is not None and spec.loader is not None
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def sample():
    X=np.array([[0.,0.],[0.,1.],[5.,5.],[5.,6.],[10.,10.]])
    return X,np.array([10,10,30,30,50])


def test_minkowski_l1_l2_and_shape_match_hand_values() -> None:
    X=np.array([[0.,0.],[1.,2.]]); Z=np.array([[3.,4.]])
    np.testing.assert_allclose(solution.pairwise_minkowski(X,Z,p=1),[[7],[4]])
    np.testing.assert_allclose(solution.pairwise_minkowski(X,Z,p=2),[[5],[np.sqrt(8)]])
    assert solution.pairwise_minkowski(X,Z).shape==(2,1)


def test_distance_matrix_is_symmetric_zero_diagonal() -> None:
    X,_=sample(); distances=solution.pairwise_minkowski(X,X)
    np.testing.assert_allclose(distances,distances.T); np.testing.assert_allclose(np.diag(distances),0)


def test_hamming_distance_is_fraction_of_mismatched_features() -> None:
    X=np.array([["red","round","sweet"],["green","round","sour"]]); Z=np.array([["red","long","sour"]])
    np.testing.assert_allclose(solution.pairwise_hamming(X,Z),[[2/3],[2/3]])


def test_within_cluster_sse_matches_hand_calculation_and_label_names() -> None:
    X,labels=sample(); assert solution.within_cluster_sse(X,labels)==pytest.approx(1.0)
    mapped=np.where(labels==10,"a",np.where(labels==30,"b","c"))
    assert solution.within_cluster_sse(X,mapped)==pytest.approx(1.0)


def test_silhouette_well_separated_points_positive_and_singleton_zero() -> None:
    X,labels=sample(); values=solution.silhouette_samples(X,labels)
    assert values.shape==(5,) and np.all(values[:4]>0.8)
    assert values[4]==0.0


def test_silhouette_matches_two_cluster_hand_value() -> None:
    X=np.array([[0.],[2.],[10.],[12.]]); labels=np.array([0,0,1,1])
    values=solution.silhouette_samples(X,labels)
    # 对x=0: a=2, b=(10+12)/2=11
    assert values[0]==pytest.approx(9/11)


def test_pair_confusion_counts_and_rand_match_hand_counts() -> None:
    truth=np.array([10,10,30,30,50]); pred=np.array([1,1,2,2,2])
    assert solution.pair_confusion_counts(truth,pred)=={"same_same":2,"same_diff":0,"diff_same":2,"diff_diff":6}
    assert solution.rand_index(truth,pred)==pytest.approx(.8)


def test_perfect_partition_has_rand_and_ari_one() -> None:
    truth=np.array([0,0,1,1,2]); pred=np.array([30,30,10,10,20])
    assert solution.rand_index(truth,pred)==pytest.approx(1)
    assert solution.adjusted_rand_index(truth,pred)==pytest.approx(1)


def test_ari_can_be_negative_for_worse_than_chance_partition() -> None:
    truth=np.array([0,0,0,1,1,1]); pred=np.array([0,1,1,0,0,1])
    assert solution.adjusted_rand_index(truth,pred)<0


def test_metrics_are_invariant_to_label_permutations() -> None:
    truth=np.array([0,0,1,1,2,2]); pred=np.array([0,1,1,1,2,2])
    truth_map=np.array([30,30,10,10,20,20]); pred_map=np.array([7,9,9,9,3,3])
    assert solution.rand_index(truth,pred)==solution.rand_index(truth_map,pred_map)
    assert solution.adjusted_rand_index(truth,pred)==pytest.approx(solution.adjusted_rand_index(truth_map,pred_map))


def test_all_singletons_and_single_cluster_degenerate_equal_partitions_score_one() -> None:
    assert solution.adjusted_rand_index(np.arange(4),np.arange(4)[::-1])==pytest.approx(1)
    assert solution.adjusted_rand_index(np.zeros(4),np.ones(4))==pytest.approx(1)


def test_inputs_are_not_modified() -> None:
    X,labels=sample(); X0,labels0=X.copy(),labels.copy()
    solution.silhouette_samples(X,labels)
    np.testing.assert_array_equal(X,X0); np.testing.assert_array_equal(labels,labels0)


@pytest.mark.parametrize("p",[0,.5,-1,np.inf,True])
def test_minkowski_rejects_invalid_p(p) -> None:
    with pytest.raises(ValueError): solution.pairwise_minkowski(np.ones((2,1)),np.ones((2,1)),p=p)


def test_distances_reject_bad_shapes_nonfinite_and_feature_mismatch() -> None:
    with pytest.raises(ValueError): solution.pairwise_minkowski(np.array([1.,2.]),np.ones((2,1)))
    with pytest.raises(ValueError): solution.pairwise_minkowski(np.ones((2,2)),np.ones((2,1)))
    with pytest.raises(ValueError): solution.pairwise_minkowski(np.array([[np.nan]]),np.ones((1,1)))
    with pytest.raises(ValueError): solution.pairwise_hamming(np.ones((2,2)),np.ones((2,1)))


def test_silhouette_rejects_one_cluster_and_label_mismatch() -> None:
    X=np.array([[0.],[1.]])
    with pytest.raises(ValueError): solution.silhouette_samples(X,np.array([0,0]))
    with pytest.raises(ValueError): solution.silhouette_samples(X,np.array([0]))


def test_external_metrics_reject_length_mismatch_and_nonfinite() -> None:
    with pytest.raises(ValueError): solution.rand_index(np.array([0,1]),np.array([0]))
    with pytest.raises(ValueError): solution.adjusted_rand_index(np.array([0.,np.nan]),np.array([0,1]))


def test_guided_demo_runs_and_reports_internal_external_metrics() -> None:
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "distance shape: (5, 5)" in result.stdout
    assert "SSE: 1.0" in result.stdout
    assert "pair counts: {'same_same': 2, 'same_diff': 0, 'diff_same': 2, 'diff_diff': 6}" in result.stdout
    assert "Rand/ARI:" in result.stdout

