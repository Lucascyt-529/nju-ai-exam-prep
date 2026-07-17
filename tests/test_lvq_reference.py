import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TOPIC=ROOT/"watermelon_book"/"09_clustering"/"03_lvq"
spec=importlib.util.spec_from_file_location("lvq_solution",TOPIC/"reference"/"solution.py")
assert spec is not None and spec.loader is not None
solution=importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def data():
    X=np.array([[0.,0.],[0.,1.],[1.,0.],[4.,4.],[4.,5.],[5.,4.]])
    return X,np.array([0,0,0,1,1,1])


def test_squared_distances_and_nearest_shape_tie_rule() -> None:
    X=np.array([[1.,0.],[2.,2.]]); prototypes=np.array([[0.,0.],[2.,0.]])
    winners,distances=solution.nearest_prototype(X,prototypes)
    np.testing.assert_allclose(distances,[[1,1],[8,4]])
    np.testing.assert_array_equal(winners,[0,1]); assert distances.shape==(2,2)


def test_correct_winner_moves_toward_sample() -> None:
    prototypes=np.array([[0.,0.],[4.,4.]]); labels=np.array([0,1]); x=np.array([1.,0.])
    updated,winner,correct=solution.lvq_update(prototypes,labels,x,0,.5)
    assert winner==0 and correct is True
    np.testing.assert_allclose(updated,[[.5,0],[4,4]])
    assert np.linalg.norm(updated[0]-x)<np.linalg.norm(prototypes[0]-x)


def test_wrong_winner_moves_away_from_sample() -> None:
    prototypes=np.array([[0.,0.],[4.,4.]]); labels=np.array([0,1]); x=np.array([1.,0.])
    updated,winner,correct=solution.lvq_update(prototypes,labels,x,1,.5)
    assert winner==0 and correct is False
    np.testing.assert_allclose(updated,[[-.5,0],[4,4]])
    assert np.linalg.norm(updated[0]-x)>np.linalg.norm(prototypes[0]-x)


def test_only_winning_prototype_changes_and_inputs_are_not_modified() -> None:
    prototypes=np.array([[0.,0.],[4.,4.]]); original=prototypes.copy(); labels=np.array([0,1]); x=np.array([1.,0.])
    updated,_,_=solution.lvq_update(prototypes,labels,x,0,.25)
    np.testing.assert_array_equal(updated[1],original[1]); np.testing.assert_array_equal(prototypes,original)


def test_stratified_initialization_has_requested_count_and_labels() -> None:
    X,y=data(); prototypes,labels,indices=solution.initialize_prototypes(X,y,2,random_state=3)
    assert prototypes.shape==(4,2); np.testing.assert_array_equal(labels,[0,0,1,1])
    assert np.all(y[indices]==labels) and len(np.unique(indices))==4


def test_initialization_is_reproducible_and_seed_can_change_indices() -> None:
    X,y=data(); first=solution.initialize_prototypes(X,y,1,random_state=1); second=solution.initialize_prototypes(X,y,1,random_state=1); third=solution.initialize_prototypes(X,y,1,random_state=2)
    np.testing.assert_array_equal(first[2],second[2]); assert not np.array_equal(first[2],third[2])


def test_fit_history_shapes_labels_and_training_prediction() -> None:
    X,y=data(); model=solution.fit_lvq(X,y,epochs=10,random_state=3)
    assert model["history"].shape==(11,2,2); assert model["correct_updates"].shape==(10,)
    np.testing.assert_array_equal(model["prototype_labels"],[0,1])
    np.testing.assert_array_equal(solution.predict(model,X),y)


def test_noncontiguous_labels_are_preserved() -> None:
    X,y=data(); labels=np.where(y==0,10,30); model=solution.fit_lvq(X,labels,epochs=5,random_state=3)
    np.testing.assert_array_equal(model["prototype_labels"],[10,30]); np.testing.assert_array_equal(solution.predict(model,X),labels)


def test_decay_one_uses_fixed_rate_and_history_has_independent_copies() -> None:
    X,y=data(); model=solution.fit_lvq(X,y,epochs=2,learning_rate=.1,decay=1,random_state=3)
    before=model["history"].copy(); model["prototypes"][0,0]+=100
    np.testing.assert_array_equal(model["history"],before)


def test_fit_is_deterministic_and_does_not_modify_inputs() -> None:
    X,y=data(); X0,y0=X.copy(),y.copy(); first=solution.fit_lvq(X,y,random_state=7); second=solution.fit_lvq(X,y,random_state=7)
    np.testing.assert_array_equal(first["history"],second["history"]); np.testing.assert_array_equal(X,X0); np.testing.assert_array_equal(y,y0)


@pytest.mark.parametrize("rate",[0,-.1,1.1,np.inf,True])
def test_update_rejects_invalid_learning_rate(rate) -> None:
    with pytest.raises(ValueError): solution.lvq_update(np.ones((2,1)),np.array([0,1]),np.array([1.]),0,rate)


def test_initialization_rejects_too_many_prototypes_and_single_class() -> None:
    X,y=data()
    with pytest.raises(ValueError): solution.initialize_prototypes(X,y,4,0)
    with pytest.raises(ValueError): solution.fit_lvq(X,np.zeros(6,dtype=int))


def test_functions_reject_bad_shapes_feature_mismatch_and_query() -> None:
    with pytest.raises(ValueError): solution.nearest_prototype(np.array([1.,2.]),np.ones((2,2)))
    with pytest.raises(ValueError): solution.lvq_update(np.ones((2,2)),np.array([0]),np.ones(2),0,.1)
    X,y=data(); model=solution.fit_lvq(X,y)
    with pytest.raises(ValueError): solution.predict(model,np.ones((2,3)))


def test_guided_demo_runs_and_reports_both_update_directions() -> None:
    result=subprocess.run([sys.executable,str(TOPIC/"guided_demo.py")],cwd=ROOT,check=True,capture_output=True,text=True,encoding="utf-8",env={**os.environ,"PYTHONUTF8":"1"})
    assert "winner/correct: 0 True" in result.stdout
    assert "toward: [[0.5, 0.0], [4.0, 4.0]]" in result.stdout
    assert "away: [[-0.5, 0.0], [4.0, 4.0]] False" in result.stdout
    assert "training prediction: [0, 0, 0, 1, 1, 1]" in result.stdout

