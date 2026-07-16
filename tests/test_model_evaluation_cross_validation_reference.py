import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "watermelon_book" / "02_model_evaluation_selection" / "03_cross_validation" / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("evaluation_cv_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def make_three_folds() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
    ]


def fit_mean_regressor(X_train: np.ndarray, y_train: np.ndarray) -> float:
    return float(y_train.mean())


def predict_mean(model: object, X_validation: np.ndarray) -> np.ndarray:
    return np.full(X_validation.shape[0], float(model))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def test_cross_validation_scores_and_oof_alignment() -> None:
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    folds = make_three_folds()
    scores = solution.cross_validation_scores(
        X, y, folds, fit_mean_regressor, predict_mean, mse
    )
    oof = solution.out_of_fold_predictions(
        X, y, folds, fit_mean_regressor, predict_mean
    )

    np.testing.assert_allclose(scores, [37.0, 1.0, 37.0])
    np.testing.assert_allclose(oof, [7.0, 7.0, 5.0, 5.0, 3.0, 3.0])
    assert mse(y, oof) == pytest.approx(25.0)


def test_fit_receives_only_current_training_fold() -> None:
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float)
    seen_training_values: list[set[float]] = []

    def recording_fit(X_train: np.ndarray, y_train: np.ndarray) -> float:
        seen_training_values.append(set(y_train.tolist()))
        return float(y_train.mean())

    solution.cross_validation_scores(
        X, y, make_three_folds(), recording_fit, predict_mean, mse
    )
    assert seen_training_values == [
        {2.0, 3.0, 4.0, 5.0},
        {0.0, 1.0, 4.0, 5.0},
        {0.0, 1.0, 2.0, 3.0},
    ]


def test_score_summary_and_candidate_direction() -> None:
    summary = solution.summarize_scores(np.array([1.0, 2.0, 3.0]))
    best_loss, loss_summaries = solution.select_best_candidate(
        {"simple": np.array([2.0, 2.0]), "complex": np.array([1.0, 1.5])},
        higher_is_better=False,
    )
    best_accuracy, _ = solution.select_best_candidate(
        {"a": np.array([0.7, 0.8]), "b": np.array([0.8, 0.9])},
        higher_is_better=True,
    )
    assert summary == pytest.approx({"mean": 2.0, "std": np.sqrt(2 / 3)})
    assert best_loss == "complex"
    assert loss_summaries["complex"]["mean"] == pytest.approx(1.25)
    assert best_accuracy == "b"


@pytest.mark.parametrize(
    "folds",
    [
        [(np.array([0, 1]), np.array([1, 2]))],
        [(np.array([1, 2]), np.array([0]))],
        [
            (np.array([2, 3]), np.array([0, 1])),
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([0, 1]), np.array([2, 3])),
        ],
    ],
)
def test_invalid_fold_definitions_are_rejected(folds) -> None:
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.arange(4, dtype=float)
    with pytest.raises((ValueError, IndexError)):
        solution.cross_validation_scores(
            X, y, folds, fit_mean_regressor, predict_mean, mse
        )


def test_bad_prediction_shape_and_metric_are_rejected() -> None:
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float)
    folds = make_three_folds()

    with pytest.raises(ValueError, match="predict"):
        solution.cross_validation_scores(
            X,
            y,
            folds,
            fit_mean_regressor,
            lambda model, values: np.array([float(model)]),
            mse,
        )
    with pytest.raises(ValueError, match="metric"):
        solution.cross_validation_scores(
            X,
            y,
            folds,
            fit_mean_regressor,
            predict_mean,
            lambda true, pred: np.nan,
        )
