import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "integrated_tasks" / "05_mixed_feature_classification"
SOLUTION = TOPIC / "reference" / "solution.py"
STARTER = TOPIC / "starter.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(SOLUTION, "mixed_feature_solution")
starter = load_module(STARTER, "mixed_feature_starter")


def train_states(tmp_path: Path, validation_path: Path | None = None):
    encoder = tmp_path / "state" / "encoder.json"
    model = tmp_path / "state" / "model.data"
    metrics = tmp_path / "output" / "metrics.txt"
    solution.run_training(
        TOPIC / "data" / "train.csv",
        validation_path or TOPIC / "data" / "validation.csv",
        encoder,
        model,
        metrics,
        learning_rate=0.2,
        n_steps=1000,
        l2=0.05,
        threshold=0.5,
    )
    return encoder, model, metrics


def assert_model_states_equal(first_path: Path, second_path: Path) -> None:
    first = solution.load_model_bundle(first_path)
    second = solution.load_model_bundle(second_path)
    for first_value, second_value in zip(first[:4], second[:4], strict=True):
        np.testing.assert_array_equal(first_value, second_value)
    assert first[4:] == second[4:]


def test_training_csv_keeps_numeric_missing_and_separates_feature_types() -> None:
    ids, numeric, categorical, labels = solution.load_mixed_classification_csv(
        TOPIC / "data" / "train.csv", has_label=True
    )
    assert len(ids) == 12 and len(set(ids)) == 12
    assert numeric.shape == (12, 2) and categorical.shape == (12, 2)
    assert labels is not None and labels.shape == (12,)
    assert np.isnan(numeric[2, 1]) and np.isnan(numeric[8, 0])
    assert categorical[0].tolist() == ["Nanjing", "mobile"]


def test_numeric_state_is_fitted_only_from_training_rows() -> None:
    _, numeric, _, _ = solution.load_mixed_classification_csv(
        TOPIC / "data" / "train.csv", has_label=True
    )
    fill_values, means, scales = solution.fit_numeric_preprocessor(numeric)
    np.testing.assert_allclose(fill_values, [31.181818181818183, 213.0])
    np.testing.assert_allclose(means, fill_values)
    np.testing.assert_allclose(scales, [7.12762445, 66.96392063], rtol=1e-8)


def test_test_missing_value_uses_saved_training_fill_value() -> None:
    X_train = np.array([[1.0, 10.0], [3.0, 30.0]])
    fill_values, means, scales = solution.fit_numeric_preprocessor(X_train)
    X_test = np.array([[np.nan, 1000.0]])
    transformed = solution.transform_numeric(X_test, fill_values, means, scales)
    assert transformed.shape == (1, 2)
    assert transformed[0, 0] == 0.0
    assert transformed[0, 1] == pytest.approx(98.0)


def test_vocabularies_are_training_only_and_unknowns_use_last_columns() -> None:
    _, _, train_categorical, _ = solution.load_mixed_classification_csv(
        TOPIC / "data" / "train.csv", has_label=True
    )
    _, _, validation_categorical, _ = solution.load_mixed_classification_csv(
        TOPIC / "data" / "validation.csv", has_label=True
    )
    vocabularies = solution.fit_vocabularies(train_categorical)
    assert vocabularies == (("Nanjing", "Suzhou"), ("desktop", "mobile"))
    encoded = solution.transform_one_hot(validation_categorical, vocabularies)
    assert encoded.shape == (4, 6)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.full(4, 2.0))
    np.testing.assert_array_equal(encoded[-1], [0, 0, 1, 0, 0, 1])
    assert "Hangzhou" not in vocabularies[0] and "tablet" not in vocabularies[1]


def test_design_matrix_preserves_numeric_then_categorical_column_blocks() -> None:
    _, numeric, categorical, _ = solution.load_mixed_classification_csv(
        TOPIC / "data" / "train.csv", has_label=True
    )
    numeric_state = solution.fit_numeric_preprocessor(numeric)
    vocabularies = solution.fit_vocabularies(categorical)
    design = solution.build_design_matrix(
        numeric, categorical, *numeric_state, vocabularies
    )
    assert design.shape == (12, 8)
    np.testing.assert_allclose(design[:, :2].mean(axis=0), [0.0, 0.0], atol=1e-15)
    np.testing.assert_array_equal(design[:, 2:5].sum(axis=1), np.ones(12))
    np.testing.assert_array_equal(design[:, 5:8].sum(axis=1), np.ones(12))


def test_logistic_training_reduces_loss_and_fits_sample_data() -> None:
    _, numeric, categorical, labels = solution.load_mixed_classification_csv(
        TOPIC / "data" / "train.csv", has_label=True
    )
    assert labels is not None
    numeric_state = solution.fit_numeric_preprocessor(numeric)
    vocabularies = solution.fit_vocabularies(categorical)
    design = solution.build_design_matrix(
        numeric, categorical, *numeric_state, vocabularies
    )
    weights, bias, losses = solution.fit_logistic_regression(
        design, labels, learning_rate=0.2, n_steps=1000, l2=0.05
    )
    assert losses.shape == (1001,) and losses[-1] < losses[0] * 0.5
    probabilities = solution.predict_probabilities(design, weights, bias)
    assert np.mean((probabilities >= 0.5) == labels) == 1.0


def test_encoder_json_round_trip_preserves_schema_and_utf8(tmp_path: Path) -> None:
    vocabularies = (("南京", "苏州"), ("桌面", "移动"))
    path = tmp_path / "nested" / "encoder.json"
    solution.save_encoder_metadata(path, vocabularies)
    assert solution.load_encoder_metadata(path) == vocabularies
    text = path.read_text(encoding="utf-8")
    assert "南京" in text and "\\u5357" not in text and text.endswith("\n")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(format_version=1.0),
        lambda payload: payload.update(extra="bad"),
        lambda payload: payload.update(numeric_features=["monthly_spend", "age"]),
        lambda payload: payload.update(categorical_features=["device", "city"]),
        lambda payload: payload.update(include_unknown=1),
        lambda payload: payload.update(vocabularies=[["Suzhou", "Nanjing"], ["desktop", "mobile"]]),
        lambda payload: payload.update(vocabularies=["Nanjing", "desktop"]),
    ],
)
def test_corrupt_encoder_json_is_rejected(tmp_path: Path, mutation) -> None:
    payload = {
        "format_version": 1,
        "numeric_features": ["age", "monthly_spend"],
        "categorical_features": ["city", "device"],
        "vocabularies": [["Nanjing", "Suzhou"], ["desktop", "mobile"]],
        "include_unknown": True,
    }
    mutation(payload)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        solution.load_encoder_metadata(path)


def test_model_bundle_round_trip_uses_exact_path_and_safe_shapes(tmp_path: Path) -> None:
    encoder, model, _ = train_states(tmp_path)
    assert encoder.is_file() and model.is_file()
    assert not (model.parent / "model.data.npz").exists()
    fill_values, means, scales, weights, bias, threshold = solution.load_model_bundle(model)
    assert fill_values.shape == means.shape == scales.shape == (2,)
    assert weights.shape == (8,)
    assert np.all(scales > 0) and np.isfinite(bias) and threshold == 0.5


@pytest.mark.parametrize(
    "bundle",
    [
        {
            "fill_values": np.ones(2), "means": np.ones(2),
            "scales": np.ones(2), "weights": np.ones(8), "bias": np.array(0.0),
        },
        {
            "fill_values": np.ones(2), "means": np.ones(2),
            "scales": np.ones(2), "weights": np.ones(8), "bias": np.array(0.0),
            "threshold": np.array(0.5), "extra": np.ones(1),
        },
        {
            "fill_values": np.ones(3), "means": np.ones(2),
            "scales": np.ones(2), "weights": np.ones(8), "bias": np.array(0.0),
            "threshold": np.array(0.5),
        },
        {
            "fill_values": np.ones(2), "means": np.ones(2),
            "scales": np.ones(2), "weights": np.ones(8), "bias": np.array([0.0]),
            "threshold": np.array(0.5),
        },
    ],
)
def test_corrupt_model_bundles_are_rejected(tmp_path: Path, bundle) -> None:
    path = tmp_path / "bad.npz"
    with path.open("wb") as file:
        np.savez(file, **bundle)
    with pytest.raises(ValueError):
        solution.load_model_bundle(path)


def test_object_model_array_is_not_loaded_with_pickle(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    with path.open("wb") as file:
        np.savez(
            file,
            fill_values=np.array([{"unsafe": True}], dtype=object),
            means=np.ones(2), scales=np.ones(2), weights=np.ones(8),
            bias=np.array(0.0), threshold=np.array(0.5),
        )
    with pytest.raises(ValueError):
        solution.load_model_bundle(path)


def test_encoder_model_width_mismatch_is_rejected_before_prediction(tmp_path: Path) -> None:
    encoder, model, _ = train_states(tmp_path)
    payload = json.loads(encoder.read_text(encoding="utf-8"))
    payload["vocabularies"][0].append("Wuxi")
    encoder.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="权重长度"):
        solution.run_prediction(
            TOPIC / "data" / "test.csv",
            encoder,
            model,
            tmp_path / "predictions.csv",
        )


def test_train_and_predict_commands_run_in_separate_processes_byte_exact(
    tmp_path: Path,
) -> None:
    encoder = tmp_path / "state" / "encoder.json"
    model = tmp_path / "state" / "model.npz"
    metrics = tmp_path / "output" / "metrics.txt"
    predictions = tmp_path / "output" / "predictions.csv"
    train = subprocess.run(
        [
            sys.executable, str(SOLUTION), "train",
            "--train", str(TOPIC / "data" / "train.csv"),
            "--validation", str(TOPIC / "data" / "validation.csv"),
            "--encoder", str(encoder), "--model", str(model),
            "--metrics", str(metrics),
        ],
        text=True, capture_output=True, check=False,
    )
    assert train.returncode == 0, train.stderr
    predict = subprocess.run(
        [
            sys.executable, str(SOLUTION), "predict",
            "--test", str(TOPIC / "data" / "test.csv"),
            "--encoder", str(encoder), "--model", str(model),
            "--predictions", str(predictions),
        ],
        text=True, capture_output=True, check=False,
    )
    assert predict.returncode == 0, predict.stderr
    assert metrics.read_bytes() == (TOPIC / "expected" / "metrics.txt").read_bytes()
    assert predictions.read_bytes() == (
        TOPIC / "expected" / "predictions.csv"
    ).read_bytes()


def test_validation_rows_do_not_change_fitted_encoder_or_model(tmp_path: Path) -> None:
    original_encoder, original_model, _ = train_states(tmp_path / "original")
    altered_validation = tmp_path / "altered_validation.csv"
    altered_validation.write_text(
        "sample_id,age,monthly_spend,city,device,label\n"
        "A,999,999,NewCity,new_device,1\n"
        "B,-999,-999,OtherCity,other_device,0\n",
        encoding="utf-8",
    )
    altered_encoder, altered_model, _ = train_states(
        tmp_path / "altered", altered_validation
    )
    assert original_encoder.read_bytes() == altered_encoder.read_bytes()
    assert_model_states_equal(original_model, altered_model)


def test_single_test_row_keeps_both_feature_matrices_two_dimensional(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one.csv"
    path.write_text(
        "sample_id,age,monthly_spend,city,device\nQ,30,,Nanjing,mobile\n",
        encoding="utf-8",
    )
    ids, numeric, categorical, labels = solution.load_mixed_classification_csv(
        path, has_label=False
    )
    assert ids == ["Q"] and numeric.shape == categorical.shape == (1, 2)
    assert labels is None and np.isnan(numeric[0, 1])


@pytest.mark.parametrize(
    "content, has_label, message",
    [
        ("sample_id,age,city,device,label\nA,1,Nanjing,mobile,0\n", True, "表头"),
        ("sample_id,age,monthly_spend,city,device\nA,1,2,Nanjing,mobile\nA,2,3,Suzhou,desktop\n", False, "重复"),
        ("sample_id,age,monthly_spend,city,device,label\nA,NaN,2,Nanjing,mobile,0\n", True, "NaN"),
        ("sample_id,age,monthly_spend,city,device,label\nA,1,2,,mobile,0\n", True, "不能为空"),
        ("sample_id,age,monthly_spend,city,device,label\nA,1,2,Nanjing,mobile,1.0\n", True, "0或1"),
        ("sample_id,age,monthly_spend,city,device\nA,1,2,Nanjing,mobile,extra\n", False, "字段数量"),
    ],
)
def test_invalid_mixed_csv_is_rejected(
    tmp_path: Path, content: str, has_label: bool, message: str
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        solution.load_mixed_classification_csv(path, has_label=has_label)


def test_cli_reports_corrupt_state_with_nonzero_exit(tmp_path: Path) -> None:
    bad_encoder = tmp_path / "bad.json"
    bad_encoder.write_text("{}", encoding="utf-8")
    model = tmp_path / "missing.npz"
    result = subprocess.run(
        [
            sys.executable, str(SOLUTION), "predict",
            "--test", str(TOPIC / "data" / "test.csv"),
            "--encoder", str(bad_encoder), "--model", str(model),
            "--predictions", str(tmp_path / "out.csv"),
        ],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "混合特征任务失败" in result.stderr


@pytest.mark.parametrize(
    "function_name",
    [
        "load_mixed_classification_csv",
        "fit_numeric_preprocessor",
        "transform_numeric",
        "fit_vocabularies",
        "transform_one_hot",
        "build_design_matrix",
        "fit_logistic_regression",
        "save_encoder_metadata",
        "load_encoder_metadata",
        "save_model_bundle",
        "load_model_bundle",
        "run_training",
        "run_prediction",
    ],
)
def test_student_entry_points_remain_unimplemented(function_name: str) -> None:
    function = getattr(starter, function_name)
    with pytest.raises(NotImplementedError):
        if function_name == "load_mixed_classification_csv":
            function(Path("x"), has_label=True)
        elif function_name == "fit_numeric_preprocessor":
            function(np.ones((1, 2)))
        elif function_name == "transform_numeric":
            function(np.ones((1, 2)), np.ones(2), np.ones(2), np.ones(2))
        elif function_name == "fit_vocabularies":
            function(np.array([["a", "b"]]))
        elif function_name == "transform_one_hot":
            function(np.array([["a", "b"]]), (("a",), ("b",)))
        elif function_name == "build_design_matrix":
            function(
                np.ones((1, 2)), np.array([["a", "b"]]),
                np.ones(2), np.ones(2), np.ones(2), (("a",), ("b",)),
            )
        elif function_name == "fit_logistic_regression":
            function(
                np.ones((1, 2)), np.ones(1, dtype=int),
                learning_rate=0.1, n_steps=1, l2=0.0,
            )
        elif function_name == "save_encoder_metadata":
            function(Path("x"), (("a",), ("b",)))
        elif function_name in {"load_encoder_metadata", "load_model_bundle"}:
            function(Path("x"))
        elif function_name == "save_model_bundle":
            function(
                Path("x"), np.ones(2), np.ones(2), np.ones(2),
                np.ones(6), 0.0, 0.5,
            )
        else:
            function()
