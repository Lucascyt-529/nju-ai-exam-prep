import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "08_ensemble_learning" / "04_combination_diversity"
spec = importlib.util.spec_from_file_location("perturbation_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def labels():
    return np.array([0, 0, 1, 1, 2, 2])


def test_bootstrap_index_matrix_shape_range_and_replacement():
    indices = solution.bootstrap_index_matrix(6, 4, random_state=3)
    assert indices.shape == (4, 6)
    assert np.all((indices >= 0) & (indices < 6))
    assert any(len(np.unique(row)) < 6 for row in indices)


def test_bootstrap_is_repeatable_and_seed_changes_plan():
    first = solution.bootstrap_index_matrix(8, 3, random_state=5)
    second = solution.bootstrap_index_matrix(8, 3, random_state=5)
    third = solution.bootstrap_index_matrix(8, 3, random_state=6)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_random_feature_subspaces_are_sorted_unique_and_in_range():
    subsets = solution.random_feature_subspaces(7, 5, 3, random_state=4)
    assert subsets.shape == (5, 3)
    assert np.all((subsets >= 0) & (subsets < 7))
    assert all(np.all(np.diff(row) > 0) for row in subsets)


def test_full_feature_subspace_contains_every_feature():
    subsets = solution.random_feature_subspaces(4, 3, 4, random_state=2)
    np.testing.assert_array_equal(subsets, np.tile(np.arange(4), (3, 1)))


def test_flipped_labels_change_exact_rounded_count_to_other_classes():
    y = labels(); copies = solution.flipped_label_copies(y, 4, flip_fraction=.34, random_state=8)
    assert copies.shape == (4, 6)
    np.testing.assert_array_equal(np.sum(copies != y[None, :], axis=1), [2, 2, 2, 2])
    assert set(np.unique(copies)).issubset(set(np.unique(y)))


def test_zero_flip_fraction_returns_equal_independent_copies():
    y = labels(); copies = solution.flipped_label_copies(y, 2, flip_fraction=0, random_state=1)
    np.testing.assert_array_equal(copies, np.tile(y, (2, 1)))
    copies[0, 0] = 99
    assert y[0] == 0 and copies[1, 0] == 0


def test_string_labels_can_be_flipped_without_numeric_encoding_assumption():
    y = np.array(["cat", "cat", "dog", "dog"])
    copies = solution.flipped_label_copies(y, 2, flip_fraction=.5, random_state=2)
    np.testing.assert_array_equal(np.sum(copies != y[None, :], axis=1), [2, 2])


def test_parameter_seeds_are_uint32_repeatable_and_correct_shape():
    first = solution.random_parameter_seeds(5, random_state=7)
    second = solution.random_parameter_seeds(5, random_state=7)
    assert first.shape == (5,) and first.dtype == np.uint32
    np.testing.assert_array_equal(first, second)


def test_combined_plan_has_all_four_auditable_mechanisms():
    y = labels(); plan = solution.diversity_perturbation_plan(6, 5, y, n_learners=3, subspace_size=2, flip_fraction=1/3, random_state=9)
    assert set(plan) == {"sample_indices", "feature_subspaces", "perturbed_labels", "parameter_seeds"}
    assert plan["sample_indices"].shape == (3, 6)
    assert plan["feature_subspaces"].shape == (3, 2)
    assert plan["perturbed_labels"].shape == (3, 6)
    assert plan["parameter_seeds"].shape == (3,)


def test_combined_plan_is_fully_reproducible_from_master_seed():
    y = labels()
    first = solution.diversity_perturbation_plan(6, 5, y, n_learners=3, subspace_size=2, random_state=12)
    second = solution.diversity_perturbation_plan(6, 5, y, n_learners=3, subspace_size=2, random_state=12)
    for key in first: np.testing.assert_array_equal(first[key], second[key])


def test_label_input_is_not_modified():
    y = labels(); original = y.copy()
    solution.flipped_label_copies(y, 3, flip_fraction=.5, random_state=1)
    solution.diversity_perturbation_plan(6, 4, y, n_learners=2, subspace_size=2)
    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize("name,value", [("n_samples", 0), ("n_learners", 0), ("n_features", 0), ("subspace_size", 0)])
def test_nonpositive_size_options_are_rejected(name, value):
    arguments = {"n_samples": 6, "n_features": 4, "y": labels(), "n_learners": 2, "subspace_size": 2}
    arguments[name] = value
    with pytest.raises(ValueError): solution.diversity_perturbation_plan(**arguments)


def test_subspace_larger_than_feature_count_is_rejected():
    with pytest.raises(ValueError): solution.random_feature_subspaces(3, 2, 4)


@pytest.mark.parametrize("fraction", [-.1, 1.1, np.inf, True])
def test_bad_flip_fraction_is_rejected(fraction):
    with pytest.raises(ValueError): solution.flipped_label_copies(labels(), 2, flip_fraction=fraction)


def test_bad_labels_length_single_class_and_seed_are_rejected():
    with pytest.raises(ValueError): solution.flipped_label_copies(np.zeros(4, dtype=int), 2)
    with pytest.raises(ValueError): solution.diversity_perturbation_plan(5, 3, labels(), n_learners=2, subspace_size=2)
    with pytest.raises(ValueError): solution.bootstrap_index_matrix(5, 2, random_state=1.5)


def test_guided_demo_reports_four_perturbation_shapes():
    result = subprocess.run([sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    assert "sample index shape: (3, 6)" in result.stdout
    assert "feature subspaces:" in result.stdout
    assert "flips per learner: [2, 2, 2]" in result.stdout
    assert "parameter seed shape: (3,)" in result.stdout
