import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "09_clustering" / "05_dbscan"
spec = importlib.util.spec_from_file_location("dbscan_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def data() -> np.ndarray:
    return np.array([[0.0], [0.2], [0.4], [0.6], [1.0], [3.0], [3.2], [3.4], [3.6]])


def test_pairwise_distance_shape_symmetry_diagonal_and_value() -> None:
    distances = solution.pairwise_euclidean(data())
    assert distances.shape == (9, 9)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0.0)
    assert distances[0, 3] == pytest.approx(0.6)


def test_epsilon_neighborhood_includes_self_and_boundary() -> None:
    X = np.array([[0.0], [0.5], [1.1]])
    neighborhoods = solution.epsilon_neighborhoods(X, eps=0.5)
    np.testing.assert_array_equal(np.diag(neighborhoods), np.ones(3, dtype=bool))
    assert neighborhoods[0, 1]
    assert not neighborhoods[1, 2]


def test_neighbor_counts_include_each_sample_itself() -> None:
    model = solution.fit_dbscan(data(), eps=0.21, min_samples=3)
    np.testing.assert_array_equal(model["neighbor_counts"], [2, 3, 3, 2, 1, 2, 3, 3, 2])


def test_core_border_noise_classification() -> None:
    model = solution.fit_dbscan(data(), eps=0.21, min_samples=3)
    np.testing.assert_array_equal(np.flatnonzero(model["core_mask"]), [1, 2, 6, 7])
    np.testing.assert_array_equal(np.flatnonzero(model["border_mask"]), [0, 3, 5, 8])
    np.testing.assert_array_equal(np.flatnonzero(model["noise_mask"]), [4])


def test_two_clusters_and_noise_labels_are_deterministic() -> None:
    model = solution.fit_dbscan(data(), eps=0.21, min_samples=3)
    np.testing.assert_array_equal(model["labels"], [0, 0, 0, 0, -1, 1, 1, 1, 1])
    assert model["clusters"] == ((0, 1, 2, 3), (5, 6, 7, 8))


def test_density_reachability_connects_core_chain() -> None:
    X = np.arange(0.0, 1.0, 0.2).reshape(-1, 1)
    model = solution.fit_dbscan(X, eps=0.21, min_samples=3)
    np.testing.assert_array_equal(model["labels"], np.zeros(5, dtype=int))
    np.testing.assert_array_equal(np.flatnonzero(model["core_mask"]), [1, 2, 3])


def test_all_noise_when_no_core_exists() -> None:
    X = np.array([[0.0], [2.0], [4.0]])
    model = solution.fit_dbscan(X, eps=0.5, min_samples=2)
    np.testing.assert_array_equal(model["labels"], [-1, -1, -1])
    assert model["clusters"] == ()
    assert np.all(model["noise_mask"])


def test_min_samples_one_makes_every_sample_core() -> None:
    X = np.array([[0.0], [0.1], [3.0]])
    model = solution.fit_dbscan(X, eps=0.2, min_samples=1)
    np.testing.assert_array_equal(model["labels"], [0, 0, 1])
    assert np.all(model["core_mask"])
    assert not np.any(model["border_mask"])


def test_large_eps_connects_all_samples() -> None:
    model = solution.fit_dbscan(data(), eps=4.0, min_samples=2)
    np.testing.assert_array_equal(model["labels"], np.zeros(9, dtype=int))


def test_fit_is_repeatable_and_does_not_modify_input() -> None:
    X = data()
    original = X.copy()
    first = solution.fit_dbscan(X, 0.21, 3)
    second = solution.fit_dbscan(X, 0.21, 3)
    np.testing.assert_array_equal(first["labels"], second["labels"])
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("eps", [0.0, -1.0, np.inf, np.nan, True, "0.5"])
def test_invalid_eps_is_rejected(eps) -> None:
    with pytest.raises(ValueError):
        solution.fit_dbscan(data(), eps, 3)


@pytest.mark.parametrize("min_samples", [0, -1, 1.5, True, "3"])
def test_invalid_min_samples_is_rejected(min_samples) -> None:
    with pytest.raises(ValueError):
        solution.fit_dbscan(data(), 0.21, min_samples)


@pytest.mark.parametrize(
    "X",
    [np.array([1.0, 2.0]), np.empty((0, 2)), np.array([[np.nan]]), np.array([["x"]])],
)
def test_invalid_X_is_rejected(X) -> None:
    with pytest.raises(ValueError):
        solution.fit_dbscan(X, 0.5, 2)


def test_guided_demo_runs_and_exposes_intermediate_states() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "neighborhood shape: (9, 9)" in result.stdout
    assert "core indices: [1, 2, 6, 7]" in result.stdout
    assert "border indices: [0, 3, 5, 8]" in result.stdout
    assert "noise indices: [4]" in result.stdout
