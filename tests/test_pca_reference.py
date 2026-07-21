import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "02_machine_learning" / "04_pca"
spec = importlib.util.spec_from_file_location("pca_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def rectangle() -> np.ndarray:
    return np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])


def test_fit_returns_shape_first_outputs() -> None:
    model = solution.fit_pca(rectangle(), 1)
    assert model["mean"].shape == (2,)
    assert model["covariance"].shape == (2, 2)
    assert model["components"].shape == (1, 2)
    assert model["explained_variance"].shape == (1,)


def test_mean_and_centered_columns_use_axis_zero() -> None:
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    model = solution.fit_pca(X, 2)
    np.testing.assert_allclose(model["mean"], [2.0, 20.0])
    np.testing.assert_allclose((X - model["mean"]).mean(axis=0), 0.0)


def test_covariance_matches_hand_value() -> None:
    model = solution.fit_pca(rectangle(), 2)
    np.testing.assert_allclose(model["covariance"], [[16.0 / 3.0, 0.0], [0.0, 4.0 / 3.0]])


def test_components_are_orthonormal() -> None:
    components = solution.fit_pca(rectangle(), 2)["components"]
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-12)


def test_known_principal_axis_and_deterministic_sign() -> None:
    component = solution.fit_pca(rectangle(), 1)["components"][0]
    np.testing.assert_allclose(component, [1.0, 0.0], atol=1e-12)


def test_explained_variance_and_ratio_match_rectangle() -> None:
    model = solution.fit_pca(rectangle(), 2)
    np.testing.assert_allclose(model["explained_variance"], [16.0 / 3.0, 4.0 / 3.0])
    np.testing.assert_allclose(model["explained_variance_ratio"], [0.8, 0.2])


def test_transform_shape_and_values() -> None:
    model = solution.fit_pca(rectangle(), 1)
    Z = solution.transform_pca(rectangle(), model["mean"], model["components"])
    assert Z.shape == (4, 1)
    np.testing.assert_allclose(Z[:, 0], [-2.0, -2.0, 2.0, 2.0])


def test_full_components_reconstruct_original_data() -> None:
    X = rectangle()
    model = solution.fit_pca(X, 2)
    Z = solution.transform_pca(X, model["mean"], model["components"])
    reconstructed = solution.inverse_transform_pca(Z, model["mean"], model["components"])
    np.testing.assert_allclose(reconstructed, X, atol=1e-12)
    assert solution.reconstruction_mse(X, reconstructed) < 1e-24


def test_one_component_discards_smaller_variance_direction() -> None:
    X = rectangle()
    model = solution.fit_pca(X, 1)
    Z = solution.transform_pca(X, model["mean"], model["components"])
    reconstructed = solution.inverse_transform_pca(Z, model["mean"], model["components"])
    assert solution.reconstruction_mse(X, reconstructed) == pytest.approx(0.5)


def test_translation_changes_mean_but_not_covariance_or_components() -> None:
    X = rectangle()
    shifted = X + np.array([10.0, -7.0])
    first = solution.fit_pca(X, 2)
    second = solution.fit_pca(shifted, 2)
    np.testing.assert_allclose(second["mean"], first["mean"] + [10.0, -7.0])
    np.testing.assert_allclose(first["covariance"], second["covariance"])
    np.testing.assert_allclose(first["components"], second["components"])


def test_constant_data_has_zero_variance_ratio_and_zero_reconstruction_error() -> None:
    X = np.full((4, 3), 5.0)
    model = solution.fit_pca(X, 2)
    np.testing.assert_allclose(model["all_explained_variance"], 0.0)
    np.testing.assert_allclose(model["explained_variance_ratio"], 0.0)
    Z = solution.transform_pca(X, model["mean"], model["components"])
    reconstructed = solution.inverse_transform_pca(Z, model["mean"], model["components"])
    np.testing.assert_allclose(reconstructed, X)


def test_single_sample_has_zero_covariance() -> None:
    X = np.array([[2.0, 3.0]])
    model = solution.fit_pca(X, 2)
    np.testing.assert_allclose(model["covariance"], np.zeros((2, 2)))
    np.testing.assert_allclose(model["explained_variance_ratio"], 0.0)


def test_fit_and_transform_do_not_modify_inputs() -> None:
    X = rectangle()
    original = X.copy()
    model = solution.fit_pca(X, 1)
    solution.transform_pca(X, model["mean"], model["components"])
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("n_components", [0, -1, 3, 1.5, True])
def test_invalid_component_count_is_rejected(n_components) -> None:
    with pytest.raises(ValueError):
        solution.fit_pca(rectangle(), n_components)


def test_bad_X_is_rejected() -> None:
    for X in (np.array([1.0, 2.0]), np.empty((0, 2)), np.array([[np.nan, 1.0]]), np.array([["x"]])):
        with pytest.raises(ValueError):
            solution.fit_pca(X, 1)


def test_transform_rejects_feature_and_parameter_shape_errors() -> None:
    X = rectangle()
    model = solution.fit_pca(X, 1)
    with pytest.raises(ValueError):
        solution.transform_pca(np.ones((2, 3)), model["mean"], model["components"])
    with pytest.raises(ValueError):
        solution.transform_pca(X, np.array([[0.0, 0.0]]), model["components"])
    with pytest.raises(ValueError):
        solution.transform_pca(X, model["mean"], np.ones((1, 3)))


def test_inverse_transform_rejects_latent_dimension_mismatch() -> None:
    model = solution.fit_pca(rectangle(), 1)
    with pytest.raises(ValueError):
        solution.inverse_transform_pca(np.ones((4, 2)), model["mean"], model["components"])


def test_reconstruction_mse_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        solution.reconstruction_mse(np.ones((3, 2)), np.ones((3, 1)))


def test_guided_demo_runs_and_reports_shapes_and_information_loss() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "reference_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "X shape: (4, 2)" in result.stdout
    assert "mean shape/value: (2,) [0.0, 0.0]" in result.stdout
    assert "components shape/value: (1, 2) [[1.0, 0.0]]" in result.stdout
    assert "Z shape: (4, 1)" in result.stdout
    assert "explained variance ratio: [0.8]" in result.stdout
    assert "reconstruction mse: 0.5" in result.stdout
