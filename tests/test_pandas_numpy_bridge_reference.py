import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "04_pandas_basics" / "05_pandas_numpy_bridge"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("pandas_numpy_bridge_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()
FEATURES = ["study_hours", "attendance"]


def load_frames():
    return pd.read_csv(TOPIC / "data" / "train.csv"), pd.read_csv(
        TOPIC / "data" / "validation.csv"
    )


def test_extract_arrays_has_explicit_order_dtype_and_shapes() -> None:
    train, _ = load_frames()
    X, y = solution.extract_supervised_arrays(train, FEATURES, "label")
    assert X.shape == (3, 2) and X.dtype == float
    assert y.shape == (3,) and np.issubdtype(y.dtype, np.integer)
    np.testing.assert_allclose(X[:, 0], [1, 3, 5])
    reversed_X, _ = solution.extract_supervised_arrays(train, FEATURES[::-1], "label")
    np.testing.assert_allclose(reversed_X[:, 0], X[:, 1])


def test_pandas_ddof_zero_statistics_match_numpy() -> None:
    train, _ = load_frames()
    X, _ = solution.extract_supervised_arrays(train, FEATURES, "label")
    means, scales = solution.fit_table_standardizer(train, FEATURES)
    np.testing.assert_allclose(means.to_numpy(), X.mean(axis=0))
    np.testing.assert_allclose(scales.to_numpy(), X.std(axis=0))
    assert not np.allclose(scales.to_numpy(), train[FEATURES].std().to_numpy())


def test_validation_uses_training_statistics_without_refitting() -> None:
    train, validation = load_frames()
    means, scales = solution.fit_table_standardizer(train, FEATURES)
    X_train = solution.transform_features(train, FEATURES, means, scales)
    X_validation = solution.transform_features(validation, FEATURES, means, scales)
    np.testing.assert_allclose(X_train.mean(axis=0), [0, 0], atol=1e-12)
    np.testing.assert_allclose(X_train.std(axis=0), [1, 1], atol=1e-12)
    assert not np.allclose(X_validation.mean(axis=0), [0, 0])
    np.testing.assert_allclose(means.to_numpy(), [3.0, 0.8])


def test_constant_feature_uses_unit_scale() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "constant": [5, 5, 5]})
    means, scales = solution.fit_table_standardizer(frame, ["x", "constant"])
    transformed = solution.transform_features(frame, ["x", "constant"], means, scales)
    assert scales["constant"] == 1.0
    np.testing.assert_array_equal(transformed[:, 1], [0, 0, 0])


def test_prediction_vector_and_strict_output(tmp_path: Path) -> None:
    _, validation = load_frames()
    frame = solution.prediction_frame(
        validation["sample_id"], np.array([0.25, 0.75])
    )
    output = tmp_path / "nested" / "predictions.csv"
    solution.save_predictions(output, frame)
    assert output.read_bytes() == (TOPIC / "expected" / "predictions.csv").read_bytes()


def test_functions_do_not_modify_dataframes_or_parameters() -> None:
    train, validation = load_frames()
    means, scales = solution.fit_table_standardizer(train, FEATURES)
    copies = train.copy(deep=True), validation.copy(deep=True), means.copy(), scales.copy()
    solution.extract_supervised_arrays(train, FEATURES, "label")
    solution.transform_features(validation, FEATURES, means, scales)
    pd.testing.assert_frame_equal(train, copies[0])
    pd.testing.assert_frame_equal(validation, copies[1])
    pd.testing.assert_series_equal(means, copies[2])
    pd.testing.assert_series_equal(scales, copies[3])


@pytest.mark.parametrize(
    "call",
    [
        lambda: solution.extract_supervised_arrays(
            pd.DataFrame({"x": [1], "label": [0]}), ["missing"], "label"
        ),
        lambda: solution.extract_supervised_arrays(
            pd.DataFrame({"x": [1], "label": [0.5]}), ["x"], "label"
        ),
        lambda: solution.extract_supervised_arrays(
            pd.DataFrame({"x": [1], "label": [0]}), ["label"], "label"
        ),
        lambda: solution.fit_table_standardizer(
            pd.DataFrame({"x": [1, np.nan]}), ["x"]
        ),
        lambda: solution.transform_features(
            pd.DataFrame({"x": [1]}),
            ["x"],
            pd.Series({"other": 0.0}),
            pd.Series({"other": 1.0}),
        ),
        lambda: solution.prediction_frame(
            pd.Series(["A", "B"]), np.array([[0.1], [0.2]])
        ),
        lambda: solution.prediction_frame(pd.Series(["A"]), np.array([np.inf])),
    ],
)
def test_invalid_bridge_inputs_are_rejected(call) -> None:
    with pytest.raises((ValueError, TypeError)):
        call()
