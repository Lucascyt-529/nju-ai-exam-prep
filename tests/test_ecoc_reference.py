import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "03_linear_models" / "04_multiclass_reduction"
SOLUTION = TOPIC / "reference" / "solution.py"
STARTER = TOPIC / "starter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(SOLUTION, "ecoc_solution")
starter = load_module(STARTER, "ecoc_starter")


def sample_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-2.0, 0.0], [-1.5, 0.5], [-2.5, -0.5],
            [2.0, 0.0], [1.5, 0.5], [2.5, -0.5],
            [0.0, 3.0], [0.5, 2.5], [-0.5, 2.5],
        ]
    )
    y = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30])
    return X, y


def long_binary_code() -> np.ndarray:
    return np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [-1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, -1, 1, -1, 1],
            [-1, -1, -1, 1, -1, 1, 1],
        ]
    )


def test_ovr_is_a_binary_ecoc_matrix() -> None:
    matrix = solution.make_ovr_coding_matrix(3)
    np.testing.assert_array_equal(
        matrix,
        [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    )
    assert matrix.shape == (3, 3) and not np.any(matrix == 0)


def test_ovo_is_a_ternary_ecoc_matrix_with_inactive_classes() -> None:
    matrix = solution.make_ovo_coding_matrix(3)
    np.testing.assert_array_equal(
        matrix,
        [[-1, -1, 0], [1, 0, -1], [0, 1, 1]],
    )
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(np.count_nonzero(matrix, axis=0), [2, 2, 2])


def test_four_class_ovo_has_six_columns_and_two_active_classes_each() -> None:
    matrix = solution.make_ovo_coding_matrix(4)
    assert matrix.shape == (4, 6)
    np.testing.assert_array_equal(np.count_nonzero(matrix, axis=0), np.full(6, 2))
    np.testing.assert_array_equal(np.sum(matrix == 1, axis=0), np.ones(6))
    np.testing.assert_array_equal(np.sum(matrix == -1, axis=0), np.ones(6))


@pytest.mark.parametrize("n_classes", [2, 0, -1, 3.0, True])
def test_invalid_class_counts_are_rejected(n_classes) -> None:
    with pytest.raises(ValueError):
        solution.make_ovr_coding_matrix(n_classes)


def test_ternary_training_targets_exclude_zero_code_classes() -> None:
    _, y = sample_data()
    classes = np.array([10, 20, 30])
    matrix = solution.make_ovo_coding_matrix(3)
    selected, targets = solution.ecoc_training_targets(y, classes, matrix, 0)
    np.testing.assert_array_equal(
        selected,
        [True, True, True, True, True, True, False, False, False],
    )
    np.testing.assert_array_equal(targets, [0, 0, 0, 1, 1, 1])


def test_ecoc_with_ovr_code_matches_existing_ovr_model() -> None:
    X, y = sample_data()
    classes, matrix, weights, biases = solution.fit_ecoc(
        X, y, solution.make_ovr_coding_matrix(3)
    )
    old_classes, old_weights, old_biases = solution.fit_ovr(X, y)
    np.testing.assert_array_equal(classes, old_classes)
    np.testing.assert_array_equal(matrix, solution.make_ovr_coding_matrix(3))
    np.testing.assert_allclose(weights, old_weights)
    np.testing.assert_allclose(biases, old_biases)
    np.testing.assert_array_equal(
        solution.predict_ecoc(X, classes, matrix, weights, biases), y
    )


def test_ecoc_with_ovo_code_matches_existing_pairwise_models_and_prediction() -> None:
    X, y = sample_data()
    matrix = solution.make_ovo_coding_matrix(3)
    classes, fitted_matrix, weights, biases = solution.fit_ecoc(X, y, matrix)
    old_classes, _, old_weights, old_biases = solution.fit_ovo(X, y)
    np.testing.assert_array_equal(classes, old_classes)
    np.testing.assert_array_equal(fitted_matrix, matrix)
    np.testing.assert_allclose(weights, old_weights)
    np.testing.assert_allclose(biases, old_biases)
    scores = solution.decision_function_ecoc(X, matrix, weights, biases)
    assert scores.shape == (9, 3)
    np.testing.assert_array_equal(
        solution.predict_ecoc(X, classes, matrix, weights, biases), y
    )


def test_fit_ecoc_does_not_modify_training_data_or_coding_matrix() -> None:
    X, y = sample_data()
    matrix = solution.make_ovo_coding_matrix(3)
    original_X, original_y, original_matrix = X.copy(), y.copy(), matrix.copy()
    _, fitted_matrix, _, _ = solution.fit_ecoc(X, y, matrix)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(matrix, original_matrix)
    assert not np.shares_memory(fitted_matrix, matrix)


def test_hard_codes_use_positive_one_for_exact_zero_score() -> None:
    scores = np.array([[-2.0, 0.0, 3.0]])
    np.testing.assert_array_equal(solution.hard_ecoc_codes(scores), [[-1, 1, 1]])


def test_hamming_and_euclidean_distances_are_computed_to_each_class_row() -> None:
    matrix = np.array([[1, 1, -1], [-1, 1, 1], [0, -1, 1]])
    predicted = np.array([[1, -1, 1]])
    hamming = solution.ecoc_distances(predicted, matrix, metric="hamming")
    euclidean = solution.ecoc_distances(predicted, matrix, metric="euclidean")
    np.testing.assert_array_equal(hamming, [[2.0, 2.0, 1.0]])
    np.testing.assert_allclose(euclidean, [[np.sqrt(8), np.sqrt(8), 1.0]])


def test_decode_returns_original_labels_and_ties_choose_first_sorted_class() -> None:
    classes = np.array([10, 20, 30])
    matrix = np.array([[1, 1], [1, -1], [-1, 1]])
    predicted = np.array([[1, 1], [-1, -1]])
    decoded = solution.decode_ecoc(predicted, classes, matrix)
    np.testing.assert_array_equal(decoded, [10, 20])


def test_long_binary_code_corrects_one_flipped_bit_in_example() -> None:
    matrix = long_binary_code()
    classes = np.array([10, 20, 30, 40])
    corrupted = matrix[2].copy()
    corrupted[0] *= -1
    assert solution.minimum_hamming_distance(matrix) == 4
    assert solution.binary_error_correction_capacity(matrix) == 1
    np.testing.assert_array_equal(
        solution.decode_ecoc(corrupted.reshape(1, -1), classes, matrix), [30]
    )


def test_two_flipped_bits_are_not_claimed_as_guaranteed_correction() -> None:
    matrix = long_binary_code()
    classes = np.array([10, 20, 30, 40])
    corrupted = matrix[2].copy()
    corrupted[[0, 2]] *= -1
    assert solution.binary_error_correction_capacity(matrix) == 1
    distances = solution.ecoc_distances(corrupted.reshape(1, -1), matrix)
    assert np.min(distances) <= 2
    assert solution.decode_ecoc(corrupted.reshape(1, -1), classes, matrix).shape == (1,)


def test_binary_correction_formula_rejects_ternary_code() -> None:
    with pytest.raises(ValueError, match="二元码"):
        solution.binary_error_correction_capacity(solution.make_ovo_coding_matrix(3))


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1, -1], [-1, 2], [1, 1]]),
        np.array([[1, 1], [1, -1], [1, 0]]),
        np.array([[1, -1], [1, -1], [-1, 1]]),
        np.ones((3, 1), dtype=int),
        np.empty((3, 0), dtype=int),
        np.array([[True, False], [False, True], [True, True]]),
    ],
)
def test_invalid_coding_matrices_are_rejected(matrix: np.ndarray) -> None:
    X, y = sample_data()
    with pytest.raises(ValueError):
        solution.fit_ecoc(X, y, matrix)


def test_unknown_label_and_invalid_classifier_index_are_rejected() -> None:
    classes = np.array([10, 20, 30])
    matrix = solution.make_ovo_coding_matrix(3)
    with pytest.raises(ValueError, match="不存在"):
        solution.ecoc_training_targets(np.array([10, 40]), classes, matrix, 0)
    with pytest.raises(ValueError, match="范围"):
        solution.ecoc_training_targets(np.array([10, 20]), classes, matrix, True)


@pytest.mark.parametrize(
    "predicted, metric",
    [
        (np.array([[1, 0, -1]]), "hamming"),
        (np.array([1, -1, 1]), "hamming"),
        (np.array([[1, -1]]), "hamming"),
        (np.array([[1, -1, 1]]), "cosine"),
    ],
)
def test_invalid_predictions_or_metrics_are_rejected(predicted, metric) -> None:
    matrix = solution.make_ovr_coding_matrix(3)
    with pytest.raises(ValueError):
        solution.ecoc_distances(predicted, matrix, metric=metric)


def test_invalid_ecoc_model_shapes_are_rejected() -> None:
    X = np.ones((2, 2))
    classes = np.array([10, 20, 30])
    matrix = solution.make_ovr_coding_matrix(3)
    with pytest.raises(ValueError, match="weights"):
        solution.predict_ecoc(
            X, classes, matrix, np.ones((2, 2)), np.zeros(3)
        )
    with pytest.raises(ValueError, match="biases"):
        solution.predict_ecoc(
            X, classes, matrix, np.ones((3, 2)), np.zeros(2)
        )


@pytest.mark.parametrize(
    "function_name",
    [
        "make_ovr_coding_matrix",
        "make_ovo_coding_matrix",
        "ecoc_training_targets",
        "fit_ecoc",
        "decision_function_ecoc",
        "hard_ecoc_codes",
        "ecoc_distances",
        "decode_ecoc",
        "predict_ecoc",
        "minimum_hamming_distance",
        "binary_error_correction_capacity",
    ],
)
def test_student_ecoc_entry_points_remain_unimplemented(function_name: str) -> None:
    function = getattr(starter, function_name)
    X, y = sample_data()
    classes = np.array([10, 20, 30])
    matrix = solution.make_ovr_coding_matrix(3)
    with pytest.raises(NotImplementedError):
        if function_name in {"make_ovr_coding_matrix", "make_ovo_coding_matrix"}:
            function(3)
        elif function_name == "ecoc_training_targets":
            function(y, classes, matrix, 0)
        elif function_name == "fit_ecoc":
            function(X, y, matrix)
        elif function_name == "decision_function_ecoc":
            function(X, matrix, np.ones((3, 2)), np.zeros(3))
        elif function_name == "hard_ecoc_codes":
            function(np.ones((1, 3)))
        elif function_name == "ecoc_distances":
            function(np.ones((1, 3), dtype=int), matrix)
        elif function_name == "decode_ecoc":
            function(np.ones((1, 3), dtype=int), classes, matrix)
        elif function_name == "predict_ecoc":
            function(X, classes, matrix, np.ones((3, 2)), np.zeros(3))
        else:
            function(matrix)


def test_guided_demo_shows_ternary_code_and_one_bit_correction() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "OvO ternary coding matrix:" in result.stdout
    assert "minimum Hamming distance: 4" in result.stdout
    assert "guaranteed binary correction bits: 1" in result.stdout
    assert "decoded original label: 30" in result.stdout
