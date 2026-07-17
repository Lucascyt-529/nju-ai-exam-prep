import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "05_neural_networks" / "06_rbf_network"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("rbf_network_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.0])
    centers = X.copy()
    return X, y, centers


def test_squared_distances_match_hand_calculation() -> None:
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    centers = np.array([[0.0, 1.0], [2.0, 2.0]])
    distances = solution.squared_euclidean_distances(X, centers)
    np.testing.assert_allclose(distances, [[1.0, 8.0], [2.0, 1.0]])
    assert distances.shape == (2, 2)


def test_design_matrix_has_rbf_columns_and_bias() -> None:
    X, _, centers = sample_data()
    design = solution.rbf_design_matrix(X, centers, 0.5)
    assert design.shape == (3, 4)
    np.testing.assert_allclose(np.diag(design[:, :3]), 1.0)
    np.testing.assert_array_equal(design[:, -1], np.ones(3))


def test_design_without_bias_has_one_column_per_center() -> None:
    X, _, centers = sample_data()
    design = solution.rbf_design_matrix(X, centers, 0.5, include_bias=False)
    assert design.shape == (3, 3)


def test_smaller_width_has_more_local_response() -> None:
    X = np.array([[1.0]])
    centers = np.array([[0.0]])
    narrow = solution.rbf_design_matrix(X, centers, 0.25, include_bias=False)[0, 0]
    wide = solution.rbf_design_matrix(X, centers, 2.0, include_bias=False)[0, 0]
    assert 0.0 < narrow < wide < 1.0


def test_unregularized_output_interpolates_training_samples() -> None:
    X, y, centers = sample_data()
    model = solution.fit_rbf_output(X, y, centers, width=0.4)
    prediction = solution.predict_rbf(model, X)
    np.testing.assert_allclose(prediction, y, atol=1e-10)
    assert solution.mean_squared_error(y, prediction) < 1e-20


def test_duplicate_centers_use_least_squares_without_failure() -> None:
    X, y, _ = sample_data()
    duplicate_centers = np.array([[0.0], [0.0], [2.0]])
    model = solution.fit_rbf_output(X, y, duplicate_centers, width=0.8)
    prediction = solution.predict_rbf(model, X)
    assert np.all(np.isfinite(model["output_weights"]))
    assert np.all(np.isfinite(prediction))


def test_regularization_shrinks_nonbias_output_weights() -> None:
    X, y, centers = sample_data()
    plain = solution.fit_rbf_output(X, y, centers, width=0.8)
    regularized = solution.fit_rbf_output(X, y, centers, width=0.8, regularization=10.0)
    assert np.linalg.norm(regularized["output_weights"][:-1]) < np.linalg.norm(
        plain["output_weights"][:-1]
    )


def test_model_and_predictions_have_expected_shapes() -> None:
    X, y, centers = sample_data()
    model = solution.fit_rbf_output(X, y, centers, width=0.5)
    assert model["centers"].shape == (3, 1)
    assert model["output_weights"].shape == (4,)
    assert solution.predict_rbf(model, X).shape == (3,)


def test_fit_and_distance_functions_do_not_modify_inputs() -> None:
    X, y, centers = sample_data()
    original_X, original_y, original_centers = X.copy(), y.copy(), centers.copy()
    solution.fit_rbf_output(X, y, centers, width=0.5)
    solution.squared_euclidean_distances(X, centers)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(centers, original_centers)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.squared_euclidean_distances(np.array([1.0]), np.ones((1, 1))),
        lambda: solution.squared_euclidean_distances(np.ones((1, 2)), np.ones((1, 3))),
        lambda: solution.rbf_design_matrix(np.ones((1, 1)), np.ones((1, 1)), 0.0),
        lambda: solution.rbf_design_matrix(
            np.ones((1, 1)), np.ones((1, 1)), 1.0, include_bias=1
        ),
        lambda: solution.fit_rbf_output(
            np.ones((2, 1)), np.ones((2, 1)), np.ones((1, 1)), 1.0
        ),
        lambda: solution.fit_rbf_output(
            np.ones((2, 1)), np.ones(2), np.ones((1, 1)), 1.0, regularization=-1.0
        ),
    ],
)
def test_invalid_rbf_inputs_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_invalid_model_is_rejected() -> None:
    X, y, centers = sample_data()
    model = solution.fit_rbf_output(X, y, centers, 0.5)
    missing = dict(model)
    missing.pop("width")
    with pytest.raises(ValueError):
        solution.predict_rbf(missing, X)

    wrong_weights = dict(model)
    wrong_weights["output_weights"] = np.zeros(3)
    with pytest.raises(ValueError):
        solution.predict_rbf(wrong_weights, X)


def test_mse_rejects_column_vector_broadcasting() -> None:
    with pytest.raises(ValueError):
        solution.mean_squared_error(np.array([0.0, 1.0]), np.array([[0.0], [1.0]]))


def test_guided_demo_reports_shapes_and_interpolation() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "X / y: (3, 1) (3,)" in result.stdout
    assert "centers: (3, 1)" in result.stdout
    assert "design: (3, 4)" in result.stdout
    assert "output weights: (4,)" in result.stdout
    assert "prediction: (3,) [0. 1. 0.]" in result.stdout
