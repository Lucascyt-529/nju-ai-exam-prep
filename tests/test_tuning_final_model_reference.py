import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "watermelon_book" / "02_model_evaluation_selection" / "06_tuning_final_model"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution():
    spec = importlib.util.spec_from_file_location("tuning_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution()


def split_data():
    X_train = np.array([[-2.0], [-1.0], [0.0], [1.0]])
    y_train = X_train[:, 0] ** 2
    X_validation = np.array([[2.0], [3.0]])
    y_validation = X_validation[:, 0] ** 2
    X_test = np.array([[4.0], [5.0]])
    y_test = X_test[:, 0] ** 2
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def polynomial_fit(X, y, parameters):
    degree = int(parameters["degree"])
    design = np.column_stack([X[:, 0] ** power for power in range(degree + 1)])
    return {
        "degree": degree,
        "weights": np.linalg.lstsq(design, y, rcond=None)[0],
        "sample_count": len(X),
        "seen": tuple(X[:, 0].tolist()),
    }


def polynomial_predict(model, X):
    design = np.column_stack(
        [X[:, 0] ** power for power in range(model["degree"] + 1)]
    )
    return design @ model["weights"]


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def test_cartesian_grid_count_order_and_scalar_normalization() -> None:
    grid = solution.cartesian_parameter_grid(
        {"ridge": [np.float64(0.0), 0.1, 1.0], "degree": [1, 2]}
    )
    assert len(grid) == 6
    assert grid[0] == {"degree": 1, "ridge": 0.0}
    assert grid[1] == {"degree": 1, "ridge": 0.1}
    assert grid[-1] == {"degree": 2, "ridge": 1.0}
    assert isinstance(grid[0]["ridge"], float)


def test_three_parameters_with_five_values_make_125_candidates() -> None:
    grid = solution.cartesian_parameter_grid(
        {"a": list(range(5)), "b": list(range(5)), "c": list(range(5))}
    )
    assert len(grid) == 125


def test_tuning_selects_quadratic_by_validation_mse() -> None:
    X_train, y_train, X_validation, y_validation, _, _ = split_data()
    selected, records = solution.tune_on_validation(
        X_train,
        y_train,
        X_validation,
        y_validation,
        [{"degree": 0}, {"degree": 1}, {"degree": 2}],
        polynomial_fit,
        polynomial_predict,
        mse,
        higher_is_better=False,
    )
    assert selected == {"degree": 2}
    assert len(records) == 3
    assert records[-1]["validation_score"] == pytest.approx(0.0, abs=1e-20)


def test_higher_is_better_selects_larger_score() -> None:
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0, 1])
    X_validation = np.array([[2.0], [3.0]])
    y_validation = np.array([1, 1])

    def fit(X, y, parameters):
        return int(parameters["label"])

    def predict(model, X):
        return np.full(len(X), model)

    def accuracy(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    selected, _ = solution.tune_on_validation(
        X_train,
        y_train,
        X_validation,
        y_validation,
        [{"label": 0}, {"label": 1}],
        fit,
        predict,
        accuracy,
        higher_is_better=True,
    )
    assert selected == {"label": 1}


def test_validation_tie_keeps_first_candidate() -> None:
    X_train, y_train, X_validation, y_validation, _, _ = split_data()

    def fit(X, y, parameters):
        return 0.0

    def predict(model, X):
        return np.zeros(len(X))

    selected, _ = solution.tune_on_validation(
        X_train,
        y_train,
        X_validation,
        y_validation,
        [{"choice": "first"}, {"choice": "second"}],
        fit,
        predict,
        mse,
        higher_is_better=False,
    )
    assert selected == {"choice": "first"}


def test_refit_uses_train_plus_validation_only() -> None:
    X_train, y_train, X_validation, y_validation, X_test, _ = split_data()
    model = solution.refit_selected_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        {"degree": 2},
        polynomial_fit,
    )
    assert model["sample_count"] == 6
    assert model["seen"] == (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert not set(X_test[:, 0]).intersection(model["seen"])


def test_complete_protocol_refits_and_tests_once() -> None:
    data = split_data()
    fit_seen = []
    metric_sizes = []

    def recording_fit(X, y, parameters):
        fit_seen.append(tuple(X[:, 0].tolist()))
        return polynomial_fit(X, y, parameters)

    def recording_metric(y_true, y_pred):
        metric_sizes.append(len(y_true))
        return mse(y_true, y_pred)

    result = solution.tune_refit_and_test(
        *data,
        [{"degree": 0}, {"degree": 1}, {"degree": 2}],
        recording_fit,
        polynomial_predict,
        recording_metric,
        higher_is_better=False,
    )
    assert result["selected_parameters"] == {"degree": 2}
    assert len(fit_seen) == 4
    assert all(seen == (-2.0, -1.0, 0.0, 1.0) for seen in fit_seen[:3])
    assert fit_seen[-1] == (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert metric_sizes == [2, 2, 2, 2]
    np.testing.assert_allclose(result["test_predictions"], [16.0, 25.0])
    assert result["test_score"] == pytest.approx(0.0, abs=1e-20)


def test_selected_parameters_and_records_are_copies() -> None:
    X_train, y_train, X_validation, y_validation, _, _ = split_data()
    candidates = [{"degree": 1}, {"degree": 2}]
    selected, records = solution.tune_on_validation(
        X_train,
        y_train,
        X_validation,
        y_validation,
        candidates,
        polynomial_fit,
        polynomial_predict,
        mse,
        higher_is_better=False,
    )
    selected["degree"] = 99
    records[0]["parameters"]["degree"] = 88
    assert candidates == [{"degree": 1}, {"degree": 2}]


@pytest.mark.parametrize(
    "choices",
    [
        {},
        {"": [1]},
        {"degree": []},
        {"degree": "123"},
        {"degree": [1, 1]},
        {"degree": [np.inf]},
        {"degree": [object()]},
    ],
)
def test_grid_rejects_invalid_choices(choices) -> None:
    with pytest.raises(ValueError):
        solution.cartesian_parameter_grid(choices)


def test_tuning_rejects_empty_duplicate_and_mismatched_candidates() -> None:
    X_train, y_train, X_validation, y_validation, _, _ = split_data()
    common = (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    for candidates in [[], [{"a": 1}, {"a": 1}], [{"a": 1}, {"b": 2}]]:
        with pytest.raises(ValueError):
            solution.tune_on_validation(
                *common,
                candidates,
                lambda X, y, p: 0,
                lambda model, X: np.zeros(len(X)),
                mse,
                higher_is_better=False,
            )


def test_tuning_rejects_bad_prediction_and_metric() -> None:
    X_train, y_train, X_validation, y_validation, _, _ = split_data()
    arguments = (
        X_train,
        y_train,
        X_validation,
        y_validation,
        [{"degree": 1}],
        polynomial_fit,
    )
    with pytest.raises(ValueError, match="predict"):
        solution.tune_on_validation(
            *arguments,
            lambda model, X: np.zeros((len(X), 1)),
            mse,
            higher_is_better=False,
        )
    with pytest.raises(ValueError, match="metric"):
        solution.tune_on_validation(
            *arguments,
            polynomial_predict,
            lambda y, p: np.nan,
            higher_is_better=False,
        )


def test_protocol_rejects_bad_split_shapes_values_and_feature_counts() -> None:
    X_train, y_train, X_validation, y_validation, X_test, y_test = split_data()
    candidates = [{"degree": 1}]
    tail = (candidates, polynomial_fit, polynomial_predict, mse)
    with pytest.raises(ValueError):
        solution.tune_refit_and_test(
            X_train[:, 0], y_train, X_validation, y_validation, X_test, y_test,
            *tail, higher_is_better=False,
        )
    with pytest.raises(ValueError):
        solution.tune_refit_and_test(
            X_train, y_train, np.ones((2, 2)), y_validation, X_test, y_test,
            *tail, higher_is_better=False,
        )
    X_bad = X_train.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        solution.tune_refit_and_test(
            X_bad, y_train, X_validation, y_validation, X_test, y_test,
            *tail, higher_is_better=False,
        )


def test_input_arrays_are_not_modified() -> None:
    data = split_data()
    before = tuple(array.copy() for array in data)
    solution.tune_refit_and_test(
        *data,
        [{"degree": 1}, {"degree": 2}],
        polynomial_fit,
        polynomial_predict,
        mse,
        higher_is_better=False,
    )
    for actual, expected in zip(data, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_guided_demo_runs_and_shows_final_refit() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "candidate count: 3" in result.stdout
    assert "selected: {'degree': 2}" in result.stdout
    assert "final fit sample count: 6" in result.stdout
    assert "test sample count excluded from fitting: 2" in result.stdout
