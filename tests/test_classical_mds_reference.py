import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "10_dimensionality_reduction" / "02_classical_mds"
spec = importlib.util.spec_from_file_location("mds_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def rectangle() -> np.ndarray:
    return np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]])


def test_pairwise_distance_shape_symmetry_and_hand_values() -> None:
    distances = solution.pairwise_euclidean(rectangle())
    assert distances.shape == (4, 4)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0.0)
    assert distances[0, 3] == pytest.approx(np.sqrt(5.0))


def test_double_center_matches_centered_coordinate_gram() -> None:
    X = rectangle()
    centered = X - X.mean(axis=0)
    gram = solution.double_center(solution.pairwise_euclidean(X))
    np.testing.assert_allclose(gram, centered @ centered.T, atol=1e-12)


def test_double_center_has_zero_row_and_column_means() -> None:
    gram = solution.double_center(solution.pairwise_euclidean(rectangle()))
    np.testing.assert_allclose(gram.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(gram.mean(axis=1), 0.0, atol=1e-12)


def test_full_embedding_reconstructs_rectangle_distances() -> None:
    distances = solution.pairwise_euclidean(rectangle())
    model = solution.classical_mds(distances, 2)
    reconstructed = solution.pairwise_euclidean(model["coordinates"])
    np.testing.assert_allclose(reconstructed, distances, atol=1e-10)


def test_line_is_exactly_reconstructed_in_one_dimension() -> None:
    X = np.array([[0.0], [1.0], [4.0], [7.0]])
    distances = solution.pairwise_euclidean(X)
    model = solution.classical_mds(distances, 1)
    assert model["coordinates"].shape == (4, 1)
    np.testing.assert_allclose(solution.pairwise_euclidean(model["coordinates"]), distances, atol=1e-10)


def test_coordinates_are_centered() -> None:
    distances = solution.pairwise_euclidean(rectangle())
    coordinates = solution.classical_mds(distances, 2)["coordinates"]
    np.testing.assert_allclose(coordinates.mean(axis=0), 0.0, atol=1e-12)


def test_eigenvalues_are_descending_and_known_for_rectangle() -> None:
    model = solution.classical_mds(solution.pairwise_euclidean(rectangle()), 2)
    assert np.all(np.diff(model["eigenvalues"]) <= 1e-12)
    np.testing.assert_allclose(model["eigenvalues"][:2], [4.0, 1.0], atol=1e-12)


def test_selected_values_and_coordinates_have_requested_shape() -> None:
    model = solution.classical_mds(solution.pairwise_euclidean(rectangle()), 3)
    assert model["selected_eigenvalues"].shape == (3,)
    assert model["coordinates"].shape == (4, 3)
    assert model["selected_eigenvalues"][2] == pytest.approx(0.0, abs=1e-12)


def test_reduced_embedding_has_positive_stress_but_full_is_near_zero() -> None:
    distances = solution.pairwise_euclidean(rectangle())
    one = solution.classical_mds(distances, 1)["coordinates"]
    two = solution.classical_mds(distances, 2)["coordinates"]
    assert solution.normalized_stress(distances, one) > 0.0
    assert solution.normalized_stress(distances, two) < 1e-10


def test_translation_and_rotation_of_source_do_not_change_mds_input() -> None:
    X = rectangle()
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    transformed = X @ rotation + np.array([10.0, -7.0])
    np.testing.assert_allclose(solution.pairwise_euclidean(X), solution.pairwise_euclidean(transformed))


def test_orientation_is_repeatable() -> None:
    distances = solution.pairwise_euclidean(rectangle())
    first = solution.classical_mds(distances, 2)["coordinates"]
    second = solution.classical_mds(distances, 2)["coordinates"]
    np.testing.assert_allclose(first, second)


def test_non_euclidean_distances_expose_negative_eigenvalue_but_return_finite_embedding() -> None:
    distances = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    model = solution.classical_mds(distances, 2)
    assert np.min(model["eigenvalues"]) < -1e-6
    assert np.all(np.isfinite(model["coordinates"]))


def test_single_sample_degenerate_case() -> None:
    distances = np.array([[0.0]])
    model = solution.classical_mds(distances, 1)
    np.testing.assert_allclose(model["coordinates"], [[0.0]])
    assert solution.normalized_stress(distances, model["coordinates"]) == 0.0


def test_input_distance_matrix_is_not_modified() -> None:
    distances = solution.pairwise_euclidean(rectangle())
    original = distances.copy()
    solution.classical_mds(distances, 2)
    np.testing.assert_array_equal(distances, original)


@pytest.mark.parametrize("n_components", [0, -1, 5, 1.5, True])
def test_invalid_component_count_is_rejected(n_components) -> None:
    distances = solution.pairwise_euclidean(rectangle())
    with pytest.raises(ValueError):
        solution.classical_mds(distances, n_components)


def test_nonsquare_asymmetric_negative_and_nonzero_diagonal_are_rejected() -> None:
    bad_matrices = [
        np.ones((2, 3)),
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, np.nan], [np.nan, 0.0]]),
    ]
    for matrix in bad_matrices:
        with pytest.raises(ValueError):
            solution.classical_mds(matrix, 1)


def test_stress_rejects_wrong_sample_count() -> None:
    distances = solution.pairwise_euclidean(rectangle())
    with pytest.raises(ValueError):
        solution.normalized_stress(distances, np.ones((3, 2)))


def test_guided_demo_runs_and_reports_reconstruction_quality() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "distance shape: (4, 4)" in result.stdout
    assert "gram row means: [0.0, 0.0, 0.0, 0.0]" in result.stdout
    assert "coordinates shape: (4, 2)" in result.stdout
    assert "full stress: 0.0" in result.stdout
