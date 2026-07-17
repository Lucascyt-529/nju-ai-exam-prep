import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "08_ensemble_learning" / "05_stacking"
spec = importlib.util.spec_from_file_location("stacking_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


class ColumnModel:
    def __init__(self, column: int):
        self.column = column

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.column]


class SeenSampleModel:
    """训练内样本预测1，未见样本预测0，用来暴露泄漏。"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.seen = set(X[:, 0].tolist())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([float(value in self.seen) for value in X[:, 0]])


def test_kfold_indices_cover_each_sample_exactly_once() -> None:
    folds = solution.kfold_validation_indices(11, 4, random_state=7)
    assert [len(fold) for fold in folds] == [3, 3, 3, 2]
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(11))
    assert all(len(np.intersect1d(a, b)) == 0 for i, a in enumerate(folds) for b in folds[i + 1:])


def test_kfold_is_reproducible_and_seed_changes_assignment() -> None:
    first = solution.kfold_validation_indices(20, 5, random_state=3)
    second = solution.kfold_validation_indices(20, 5, random_state=3)
    other = solution.kfold_validation_indices(20, 5, random_state=4)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))


def test_oof_predictions_do_not_let_a_sample_predict_itself() -> None:
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = X[:, 0] ** 2
    report = solution.build_oof_meta_features(X, y, [SeenSampleModel], n_splits=3)
    np.testing.assert_array_equal(report["meta_features"], np.zeros((12, 1)))
    np.testing.assert_array_equal(report["base_models"][0].predict(X), np.ones(12))


def test_oof_columns_and_original_sample_order_are_preserved() -> None:
    X = np.column_stack((np.arange(10, dtype=float), np.arange(10, dtype=float) ** 2))
    y = X[:, 0] - X[:, 1]
    report = solution.build_oof_meta_features(
        X, y, [lambda: ColumnModel(0), lambda: ColumnModel(1)],
        n_splits=4, random_state=9,
    )
    np.testing.assert_allclose(report["meta_features"], X)
    assert report["meta_features"].shape == (10, 2)


def test_ridge_combiner_recovers_hand_constructed_linear_rule() -> None:
    meta = np.array([[0., 0.], [1., 0.], [0., 1.], [2., -1.], [-1., 2.]])
    y = 4.0 + 2.0 * meta[:, 0] - 3.0 * meta[:, 1]
    combiner = solution.fit_ridge_combiner(meta, y, l2=0.0)
    assert combiner["intercept"] == pytest.approx(4.0)
    np.testing.assert_allclose(combiner["weights"], [2.0, -3.0], atol=1e-10)
    np.testing.assert_allclose(solution.predict_ridge_combiner(meta, combiner), y)


def test_complete_stacking_regression_train_predict_closure() -> None:
    X = np.array([
        [-2., 1.], [-1., -2.], [0., 2.], [1., -1.],
        [2., 3.], [3., 0.], [4., -3.], [5., 2.],
    ])
    y = 1.5 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
    model = solution.fit_stacking_regressor(
        X, y, [lambda: ColumnModel(0), lambda: ColumnModel(1)],
        n_splits=4, l2=0.0, random_state=12,
    )
    np.testing.assert_allclose(model["oof_meta_features"], X)
    np.testing.assert_allclose(solution.predict_stacking_regressor(X, model), y, atol=1e-10)


def test_regularization_does_not_penalize_constant_intercept() -> None:
    meta = np.zeros((6, 2))
    y = np.full(6, 7.5)
    combiner = solution.fit_ridge_combiner(meta, y, l2=1e6)
    assert combiner["intercept"] == pytest.approx(7.5)
    np.testing.assert_allclose(combiner["weights"], 0.0)


@pytest.mark.parametrize(
    ("n_samples", "n_splits"),
    [(0, 2), (5, 1), (5, 6), (True, 2)],
)
def test_invalid_fold_arguments_are_rejected(n_samples: int, n_splits: int) -> None:
    with pytest.raises(ValueError):
        solution.kfold_validation_indices(n_samples, n_splits)


def test_bad_base_prediction_shape_is_rejected() -> None:
    class MatrixPredictor:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros((len(X), 1))

    X = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError, match="形状"):
        solution.build_oof_meta_features(X, np.arange(6.0), [MatrixPredictor], n_splits=3)


def test_invalid_combiner_inputs_are_rejected() -> None:
    meta = np.ones((4, 2))
    y = np.arange(4.0)
    with pytest.raises(ValueError):
        solution.fit_ridge_combiner(meta, y, l2=-1)
    with pytest.raises(ValueError):
        solution.predict_ridge_combiner(meta, {"intercept": 0.0, "weights": np.ones(3)})


def test_starter_remains_unimplemented() -> None:
    starter_spec = importlib.util.spec_from_file_location("stacking_starter", TOPIC / "starter.py")
    assert starter_spec is not None and starter_spec.loader is not None
    starter = importlib.util.module_from_spec(starter_spec)
    starter_spec.loader.exec_module(starter)
    with pytest.raises(NotImplementedError):
        starter.kfold_validation_indices(4, 2)


def test_guided_demo_makes_leakage_visible() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=TOPIC, capture_output=True, text=True, encoding="utf-8", check=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "训练内预测: [1, 1, 1, 1, 1, 1, 1, 1]" in completed.stdout
    assert "折外预测:   [0, 0, 0, 0, 0, 0, 0, 0]" in completed.stdout
    assert "每个样本只出现一次: True" in completed.stdout
