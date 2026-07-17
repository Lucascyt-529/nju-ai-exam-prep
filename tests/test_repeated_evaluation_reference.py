import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "02_model_evaluation_selection" / "07_repeated_evaluation"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("repeated_evaluation_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def fit_mean_regressor(X_train: np.ndarray, y_train: np.ndarray) -> float:
    return float(y_train.mean())


def predict_mean(model: object, X_test: np.ndarray) -> np.ndarray:
    return np.full(X_test.shape[0], float(model))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0])
    return X, y


def test_seed_generation_is_reproducible_unique_and_changes_with_base_seed() -> None:
    first = solution.generate_repetition_seeds(20, 42)
    second = solution.generate_repetition_seeds(20, 42)
    changed = solution.generate_repetition_seeds(20, 43)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (20,)
    assert np.unique(first).size == 20
    assert not np.array_equal(first, changed)


@pytest.mark.parametrize("repetitions", [0, -1])
def test_invalid_repetitions_are_rejected(repetitions: int) -> None:
    with pytest.raises(ValueError, match="repetitions"):
        solution.generate_repetition_seeds(repetitions, 42)


@pytest.mark.parametrize("value", [True, 1.5, "3"])
def test_non_integer_repetitions_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="repetitions"):
        solution.generate_repetition_seeds(value, 42)


def test_invalid_base_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="base_seed"):
        solution.generate_repetition_seeds(3, -1)
    with pytest.raises(TypeError, match="base_seed"):
        solution.generate_repetition_seeds(3, True)


def test_repeated_holdout_shape_partitions_and_reproducibility(regression_data) -> None:
    X, y = regression_data
    first = solution.repeated_holdout_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=8, test_size=0.25, base_seed=7,
    )
    second = solution.repeated_holdout_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=8, test_size=0.25, base_seed=7,
    )

    assert first["scores"].shape == (8,)
    np.testing.assert_array_equal(first["scores"], second["scores"])
    np.testing.assert_array_equal(first["seeds"], second["seeds"])
    for train_indices, test_indices in zip(first["train_indices"], first["test_indices"]):
        assert train_indices.size == 9
        assert test_indices.size == 3
        assert np.intersect1d(train_indices, test_indices).size == 0
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train_indices, test_indices])), np.arange(12)
        )


def test_changed_seed_changes_at_least_one_holdout_partition(regression_data) -> None:
    X, y = regression_data
    first = solution.repeated_holdout_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=5, test_size=0.25, base_seed=7,
    )
    changed = solution.repeated_holdout_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=5, test_size=0.25, base_seed=8,
    )
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(first["test_indices"], changed["test_indices"])
    )


def test_fit_receives_only_each_holdout_training_part(regression_data) -> None:
    X, y = regression_data
    seen: list[np.ndarray] = []

    def recording_fit(X_train: np.ndarray, y_train: np.ndarray) -> float:
        seen.append(X_train[:, 0].copy())
        return float(y_train.mean())

    result = solution.repeated_holdout_scores(
        X, y, recording_fit, predict_mean, mse,
        repetitions=4, test_size=0.25, base_seed=3,
    )
    assert len(seen) == 4
    for observed, train_indices, test_indices in zip(
        seen, result["train_indices"], result["test_indices"]
    ):
        np.testing.assert_array_equal(observed, X[train_indices, 0])
        assert set(observed).isdisjoint(set(X[test_indices, 0]))


def test_stratified_holdout_keeps_every_class_in_both_parts() -> None:
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.array([0] * 14 + [1] * 6)
    result = solution.repeated_holdout_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=10, test_size=0.25, base_seed=9, stratified=True,
    )
    for train_indices, test_indices in zip(result["train_indices"], result["test_indices"]):
        assert set(y[train_indices]) == {0, 1}
        assert set(y[test_indices]) == {0, 1}
        assert np.count_nonzero(y[test_indices] == 1) == 2


def test_repeated_kfold_shape_fit_count_and_complete_coverage(regression_data) -> None:
    X, y = regression_data
    fit_calls = 0

    def counting_fit(X_train: np.ndarray, y_train: np.ndarray) -> float:
        nonlocal fit_calls
        fit_calls += 1
        return float(y_train.mean())

    result = solution.repeated_kfold_scores(
        X, y, counting_fit, predict_mean, mse,
        repetitions=3, n_splits=4, base_seed=11,
    )

    assert result["scores"].shape == (3, 4)
    assert fit_calls == 12
    for folds in result["folds"]:
        validation = np.concatenate([test for _, test in folds])
        np.testing.assert_array_equal(np.sort(validation), np.arange(12))
        for train_indices, test_indices in folds:
            assert np.intersect1d(train_indices, test_indices).size == 0
            np.testing.assert_array_equal(
                np.sort(np.concatenate([train_indices, test_indices])), np.arange(12)
            )


def test_repeated_kfold_is_reproducible(regression_data) -> None:
    X, y = regression_data
    first = solution.repeated_kfold_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=2, n_splits=3, base_seed=5,
    )
    second = solution.repeated_kfold_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=2, n_splits=3, base_seed=5,
    )
    np.testing.assert_array_equal(first["scores"], second["scores"])
    for first_repeat, second_repeat in zip(first["folds"], second["folds"]):
        for first_fold, second_fold in zip(first_repeat, second_repeat):
            np.testing.assert_array_equal(first_fold[0], second_fold[0])
            np.testing.assert_array_equal(first_fold[1], second_fold[1])


def test_stratified_kfold_keeps_classes_in_each_validation_fold() -> None:
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = np.array([0] * 12 + [1] * 12)
    result = solution.repeated_kfold_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=3, n_splits=4, base_seed=17, stratified=True,
    )
    for folds in result["folds"]:
        for _, validation_indices in folds:
            assert np.count_nonzero(y[validation_indices] == 0) == 3
            assert np.count_nonzero(y[validation_indices] == 1) == 3


def test_summary_distinguishes_run_and_repeat_statistics() -> None:
    scores = np.array([[1.0, 3.0], [5.0, 7.0]])
    summary = solution.summarize_repeated_scores(scores)

    assert summary["overall_mean"] == pytest.approx(4.0)
    assert summary["run_std"] == pytest.approx(np.std([1.0, 3.0, 5.0, 7.0]))
    np.testing.assert_allclose(summary["repeat_means"], [2.0, 6.0])
    assert summary["repeat_mean_std"] == pytest.approx(2.0)
    assert summary["fit_count"] == 4


def test_one_dimensional_summary_treats_each_holdout_as_one_repeat() -> None:
    summary = solution.summarize_repeated_scores(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(summary["repeat_means"], [1.0, 2.0, 3.0])
    assert summary["run_std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert summary["repeat_mean_std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert summary["fit_count"] == 3


@pytest.mark.parametrize(
    "scores",
    [np.array([]), np.empty((0, 2)), np.ones((2, 2, 1)), np.array([1.0, np.nan])],
)
def test_invalid_score_arrays_are_rejected(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match="scores"):
        solution.summarize_repeated_scores(scores)


def test_run_count_for_holdout_and_ten_by_ten_cv() -> None:
    assert solution.evaluation_run_count(100) == 100
    assert solution.evaluation_run_count(10, n_splits=10) == 100
    with pytest.raises(ValueError, match="n_splits"):
        solution.evaluation_run_count(10, n_splits=1)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, np.nan])
def test_invalid_holdout_size_is_rejected(regression_data, test_size: float) -> None:
    X, y = regression_data
    with pytest.raises(ValueError, match="test_size"):
        solution.repeated_holdout_scores(
            X, y, fit_mean_regressor, predict_mean, mse,
            repetitions=2, test_size=test_size, base_seed=1,
        )


def test_bad_prediction_and_metric_are_rejected(regression_data) -> None:
    X, y = regression_data
    with pytest.raises(ValueError, match="predict"):
        solution.repeated_holdout_scores(
            X, y, fit_mean_regressor, lambda model, values: np.array([model]), mse,
            repetitions=2, test_size=0.25, base_seed=1,
        )
    with pytest.raises(ValueError, match="metric"):
        solution.repeated_kfold_scores(
            X, y, fit_mean_regressor, predict_mean, lambda true, pred: np.inf,
            repetitions=2, n_splits=3, base_seed=1,
        )


def test_bad_data_and_impossible_stratification_are_rejected() -> None:
    with pytest.raises(ValueError, match="X"):
        solution.repeated_holdout_scores(
            np.array([1.0, 2.0]), np.array([0, 1]), fit_mean_regressor,
            predict_mean, mse, repetitions=2, test_size=0.5, base_seed=1,
        )
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError, match="至少需要2个样本"):
        solution.repeated_holdout_scores(
            X, y, fit_mean_regressor, predict_mean, mse,
            repetitions=2, test_size=0.3, base_seed=1, stratified=True,
        )
    with pytest.raises(ValueError, match="少于折数"):
        solution.repeated_kfold_scores(
            X, y, fit_mean_regressor, predict_mean, mse,
            repetitions=2, n_splits=3, base_seed=1, stratified=True,
        )


def test_input_arrays_are_not_modified(regression_data) -> None:
    X, y = regression_data
    X_before = X.copy()
    y_before = y.copy()
    solution.repeated_holdout_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=3, test_size=0.25, base_seed=1,
    )
    solution.repeated_kfold_scores(
        X, y, fit_mean_regressor, predict_mean, mse,
        repetitions=2, n_splits=3, base_seed=1,
    )
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_student_starter_remains_unimplemented() -> None:
    starter_text = (TOPIC / "starter.py").read_text(encoding="utf-8")
    assert starter_text.count("NotImplementedError") == 5


def test_guided_demo_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "repeated 4-fold score shape: (3, 4)" in completed.stdout
    assert "10 x 10 fit count: 100" in completed.stdout
