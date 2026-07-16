import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "02_model_evaluation_selection" / "01_data_splitting" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("evaluation_splitting_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def assert_partition(train: np.ndarray, test: np.ndarray, n_samples: int) -> None:
    assert np.intersect1d(train, test).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((train, test))), np.arange(n_samples)
    )


def test_holdout_is_disjoint_complete_and_reproducible() -> None:
    first = solution.train_test_split_indices(10, 0.2, 42)
    second = solution.train_test_split_indices(10, 0.2, 42)
    assert_partition(*first, 10)
    assert first[1].shape == (2,)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_stratified_holdout_keeps_each_class_on_both_sides() -> None:
    y = np.array([0] * 8 + [1] * 4)
    train, test = solution.stratified_train_test_split_indices(y, 0.25, 7)
    assert_partition(train, test, len(y))
    assert set(y[train]) == {0, 1}
    assert set(y[test]) == {0, 1}
    assert np.count_nonzero(y[test] == 0) == 2
    assert np.count_nonzero(y[test] == 1) == 1


def test_kfold_validation_indices_cover_every_sample_once() -> None:
    folds = solution.kfold_indices(11, 5, 42)
    validation = np.concatenate([fold[1] for fold in folds])
    np.testing.assert_array_equal(np.sort(validation), np.arange(11))
    assert len(np.unique(validation)) == 11
    for train, test in folds:
        assert_partition(train, test, 11)


def test_stratified_kfold_balances_classes() -> None:
    y = np.array([0] * 9 + [1] * 6)
    folds = solution.stratified_kfold_indices(y, 3, 42)
    for train, validation in folds:
        assert_partition(train, validation, len(y))
        assert np.count_nonzero(y[validation] == 0) == 3
        assert np.count_nonzero(y[validation] == 1) == 2


def test_bootstrap_has_expected_length_and_oob_definition() -> None:
    training, out_of_bag = solution.bootstrap_indices(20, 42)
    assert training.shape == (20,)
    expected_oob = np.setdiff1d(np.arange(20), np.unique(training))
    np.testing.assert_array_equal(out_of_bag, expected_oob)
    assert np.intersect1d(np.unique(training), out_of_bag).size == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.train_test_split_indices(1, 0.2, 0),
        lambda: solution.train_test_split_indices(5, 1.0, 0),
        lambda: solution.kfold_indices(5, 6, 0),
        lambda: solution.stratified_train_test_split_indices(np.array([0, 0, 1]), 0.2, 0),
        lambda: solution.stratified_kfold_indices(np.array([0, 0, 1, 1]), 3, 0),
    ],
)
def test_invalid_split_requests_are_rejected(call) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()
