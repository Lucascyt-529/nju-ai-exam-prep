import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "02_machine_learning" / "10_lda"
SOLUTION = TOPIC / "reference" / "solution.py"
STARTER = TOPIC / "starter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(SOLUTION, "multiclass_lda_solution")
starter = load_module(STARTER, "multiclass_lda_starter")


def sample_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-2.2, 0.2], [-2.0, 0.0], [-1.8, -0.2],
            [2.2, 0.2], [2.0, 0.0], [1.8, -0.2],
            [0.2, 3.2], [0.0, 3.0], [-0.2, 2.8],
        ]
    )
    y = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30])
    return X, y


def test_multiclass_means_and_scatter_identity_are_hand_checkable() -> None:
    X, y = sample_data()
    classes, global_mean, means, within, between, total = (
        solution.multiclass_scatter_matrices(X, y)
    )
    np.testing.assert_array_equal(classes, [10, 20, 30])
    np.testing.assert_allclose(global_mean, [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(means, [[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    for matrix in (within, between, total):
        assert matrix.shape == (2, 2)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-15)
    np.testing.assert_allclose(total, within + between, atol=1e-14)


def test_multiclass_fit_uses_at_most_k_minus_one_directions() -> None:
    X, y = sample_data()
    classes, projection, centroids, eigenvalues = solution.fit_multiclass_lda(
        X, y, regularization=1e-6
    )
    assert classes.shape == (3,)
    assert projection.shape == (2, 2)
    assert centroids.shape == (3, 2)
    assert eigenvalues.shape == (2,)
    assert np.all(np.diff(eigenvalues) <= 0) and np.all(eigenvalues > 0)
    np.testing.assert_allclose(np.linalg.norm(projection, axis=0), [1.0, 1.0])


def test_multiclass_lda_predicts_noncontiguous_original_labels() -> None:
    X, y = sample_data()
    classes, projection, centroids, _ = solution.fit_multiclass_lda(
        X, y, regularization=1e-6
    )
    projected = solution.project_multiclass(X, projection)
    assert projected.shape == (9, 2)
    prediction = solution.predict_multiclass_lda(
        X, classes, projection, centroids
    )
    np.testing.assert_array_equal(prediction, y)


def test_each_direction_satisfies_regularized_generalized_eigen_relation() -> None:
    X, y = sample_data()
    regularization = 1e-6
    _, _, _, within, between, _ = solution.multiclass_scatter_matrices(X, y)
    _, projection, _, eigenvalues = solution.fit_multiclass_lda(
        X, y, regularization=regularization
    )
    adjusted = within + regularization * np.eye(X.shape[1])
    for column, eigenvalue in enumerate(eigenvalues):
        direction = projection[:, column]
        np.testing.assert_allclose(
            between @ direction,
            eigenvalue * (adjusted @ direction),
            rtol=1e-8,
            atol=1e-8,
        )


def test_one_component_projection_is_supported() -> None:
    X, y = sample_data()
    classes, projection, centroids, eigenvalues = solution.fit_multiclass_lda(
        X, y, n_components=1, regularization=1e-6
    )
    assert projection.shape == (2, 1)
    assert centroids.shape == (3, 1)
    assert eigenvalues.shape == (1,)
    assert solution.predict_multiclass_lda(
        X, classes, projection, centroids
    ).shape == (9,)


def test_translation_preserves_scatter_and_predictions() -> None:
    X, y = sample_data()
    offset = np.array([100.0, -50.0])
    original_scatter = solution.multiclass_scatter_matrices(X, y)[3:]
    shifted_scatter = solution.multiclass_scatter_matrices(X + offset, y)[3:]
    for original, shifted in zip(original_scatter, shifted_scatter, strict=True):
        np.testing.assert_allclose(original, shifted, atol=1e-12)
    original = solution.fit_multiclass_lda(X, y, regularization=1e-6)
    shifted = solution.fit_multiclass_lda(X + offset, y, regularization=1e-6)
    np.testing.assert_allclose(original[1], shifted[1], atol=1e-12)
    np.testing.assert_array_equal(
        solution.predict_multiclass_lda(X, original[0], original[1], original[2]),
        solution.predict_multiclass_lda(
            X + offset, shifted[0], shifted[1], shifted[2]
        ),
    )


def test_collinear_class_means_have_only_one_effective_direction() -> None:
    X = np.array(
        [
            [-3.1, 0.1], [-2.9, -0.1],
            [-0.1, 0.1], [0.1, -0.1],
            [2.9, 0.1], [3.1, -0.1],
        ]
    )
    y = np.array([10, 10, 20, 20, 30, 30])
    _, projection, _, eigenvalues = solution.fit_multiclass_lda(
        X, y, regularization=1e-6
    )
    assert projection.shape == (2, 1) and eigenvalues.shape == (1,)
    with pytest.raises(ValueError, match="只有1个"):
        solution.fit_multiclass_lda(
            X, y, n_components=2, regularization=1e-6
        )


def test_zero_within_scatter_needs_regularization() -> None:
    X = np.array([[-2.0, 0.0]] * 2 + [[2.0, 0.0]] * 2 + [[0.0, 3.0]] * 2)
    y = np.array([10, 10, 20, 20, 30, 30])
    with pytest.raises(ValueError, match="增加regularization"):
        solution.fit_multiclass_lda(X, y, regularization=0.0)
    classes, projection, centroids, _ = solution.fit_multiclass_lda(
        X, y, regularization=1e-4
    )
    np.testing.assert_array_equal(
        solution.predict_multiclass_lda(X, classes, projection, centroids), y
    )


def test_duplicate_feature_is_handled_with_positive_regularization() -> None:
    X, y = sample_data()
    duplicated = np.column_stack((X, X[:, 0]))
    classes, projection, centroids, eigenvalues = solution.fit_multiclass_lda(
        duplicated, y, regularization=1e-4
    )
    assert projection.shape == (3, 2) and np.all(np.isfinite(eigenvalues))
    np.testing.assert_array_equal(
        solution.predict_multiclass_lda(
            duplicated, classes, projection, centroids
        ),
        y,
    )


def test_nearest_centroid_tie_chooses_first_sorted_original_label() -> None:
    X = np.array([[0.0]])
    classes = np.array([10, 20, 30])
    projection = np.array([[1.0]])
    centroids = np.array([[-1.0], [1.0], [3.0]])
    np.testing.assert_array_equal(
        solution.predict_multiclass_lda(X, classes, projection, centroids), [10]
    )


def test_trace_ratio_is_positive_and_projection_scale_invariant() -> None:
    X, y = sample_data()
    _, _, _, within, between, _ = solution.multiclass_scatter_matrices(X, y)
    _, projection, _, _ = solution.fit_multiclass_lda(
        X, y, n_components=1, regularization=1e-6
    )
    ratio = solution.multiclass_trace_ratio(projection, within, between)
    scaled = solution.multiclass_trace_ratio(10.0 * projection, within, between)
    assert ratio > 0 and scaled == pytest.approx(ratio)


def test_fit_does_not_modify_training_arrays() -> None:
    X, y = sample_data()
    original_X, original_y = X.copy(), y.copy()
    solution.fit_multiclass_lda(X, y, regularization=1e-6)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize(
    "X, y",
    [
        (np.ones((3, 2)), np.array([0, 0, 1])),
        (np.ones((3, 2)), np.array([0, 1, np.nan])),
        (np.ones((3, 2)), np.array([0, 1, 2]).reshape(-1, 1)),
        (np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]]), np.array([0, 1, 2])),
    ],
)
def test_invalid_multiclass_training_data_are_rejected(X, y) -> None:
    with pytest.raises(ValueError):
        solution.multiclass_scatter_matrices(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": 0},
        {"n_components": 3},
        {"n_components": True},
        {"regularization": -1.0},
        {"regularization": True},
        {"tolerance": 0.0},
        {"tolerance": np.nan},
    ],
)
def test_invalid_multiclass_hyperparameters_are_rejected(kwargs) -> None:
    X, y = sample_data()
    with pytest.raises(ValueError):
        solution.fit_multiclass_lda(X, y, **kwargs)


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.project_multiclass(np.ones((2, 2)), np.ones((3, 1))),
        lambda: solution.project_multiclass(np.ones((2, 2)), np.ones(2)),
        lambda: solution.predict_multiclass_lda(
            np.ones((1, 2)), np.array([20, 10, 30]), np.ones((2, 1)), np.ones((3, 1))
        ),
        lambda: solution.predict_multiclass_lda(
            np.ones((1, 2)), np.array([10, 20, 30]), np.ones((2, 1)), np.ones((3, 2))
        ),
        lambda: solution.multiclass_trace_ratio(
            np.ones((2, 1)), np.ones((3, 3)), np.ones((2, 2))
        ),
    ],
)
def test_invalid_projection_model_and_scatter_shapes_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()


@pytest.mark.parametrize(
    "function_name",
    [
        "multiclass_scatter_matrices",
        "fit_multiclass_lda",
        "project_multiclass",
        "predict_multiclass_lda",
        "multiclass_trace_ratio",
    ],
)
def test_student_multiclass_entry_points_remain_unimplemented(
    function_name: str,
) -> None:
    X, y = sample_data()
    function = getattr(starter, function_name)
    with pytest.raises(NotImplementedError):
        if function_name in {"multiclass_scatter_matrices", "fit_multiclass_lda"}:
            function(X, y)
        elif function_name == "project_multiclass":
            function(X, np.ones((2, 1)))
        elif function_name == "predict_multiclass_lda":
            function(X, np.array([10, 20, 30]), np.ones((2, 1)), np.ones((3, 1)))
        else:
            function(np.ones((2, 1)), np.eye(2), np.eye(2))


def test_guided_demo_shows_scatter_identity_and_dimension_limit() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "reference_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "multiclass classes: [10 20 30]" in result.stdout
    assert "St equals Sw + Sb: True" in result.stdout
    assert "projection shape (at most K-1): (2, 2)" in result.stdout
    assert "training prediction: [10 10 10 20 20 20 30 30 30]" in result.stdout
