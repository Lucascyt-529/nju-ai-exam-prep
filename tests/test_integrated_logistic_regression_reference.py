import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "integrated_tasks" / "03_logistic_regression_csv"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("integrated_logistic_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def run_pipeline(tmp_path: Path, validation_path: Path | None = None):
    predictions = tmp_path / "output" / "predictions.csv"
    metrics = tmp_path / "output" / "metrics.txt"
    bundle = tmp_path / "model" / "bundle.npz"
    solution.run_pipeline(
        TOPIC / "data" / "train.csv",
        validation_path or TOPIC / "data" / "validation.csv",
        TOPIC / "data" / "test.csv",
        predictions,
        metrics,
        bundle,
        learning_rate=0.2,
        n_steps=1000,
        l2=0.05,
        threshold=0.5,
    )
    return predictions, metrics, bundle


def test_training_standardizer_uses_known_training_statistics() -> None:
    _, X, y = solution.load_classification_csv(TOPIC / "data" / "train.csv", has_label=True)
    assert y is not None and X.shape == (8, 2) and y.shape == (8,)
    means, scales = solution.fit_standardizer(X)
    np.testing.assert_allclose(means, [0.0, 0.0])
    np.testing.assert_allclose(scales, [np.sqrt(2.5), np.sqrt(250.0)])
    transformed = solution.transform_standardizer(X, means, scales)
    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(transformed.std(axis=0), [1.0, 1.0])


def test_full_command_line_outputs_are_byte_exact(tmp_path: Path) -> None:
    predictions = tmp_path / "output" / "predictions.csv"
    metrics = tmp_path / "output" / "metrics.txt"
    bundle = tmp_path / "model" / "bundle.npz"
    completed = subprocess.run(
        [
            sys.executable, str(SOLUTION),
            "--train", str(TOPIC / "data" / "train.csv"),
            "--validation", str(TOPIC / "data" / "validation.csv"),
            "--test", str(TOPIC / "data" / "test.csv"),
            "--predictions", str(predictions), "--metrics", str(metrics),
            "--bundle", str(bundle),
        ],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert predictions.read_bytes() == (TOPIC / "expected" / "predictions.csv").read_bytes()
    assert metrics.read_bytes() == (TOPIC / "expected" / "metrics.txt").read_bytes()
    assert bundle.is_file()


def test_bundle_round_trip_contains_preprocessor_model_and_threshold(tmp_path: Path) -> None:
    _, _, bundle = run_pipeline(tmp_path)
    means, scales, weights, bias, threshold = solution.load_bundle(bundle)
    np.testing.assert_allclose(means, [0.0, 0.0])
    np.testing.assert_allclose(scales, [np.sqrt(2.5), np.sqrt(250.0)])
    np.testing.assert_allclose(weights, [1.3551291215247772] * 2)
    assert bias == pytest.approx(0.0, abs=1e-15)
    assert threshold == 0.5


def test_validation_data_does_not_change_fitted_bundle(tmp_path: Path) -> None:
    original_dir = tmp_path / "original"
    altered_dir = tmp_path / "altered"
    _, _, original_bundle = run_pipeline(original_dir)
    altered_validation = tmp_path / "validation.csv"
    altered_validation.write_text(
        "sample_id,feature_1,feature_2,label\n"
        "V1,-999,-999,1\nV2,999,999,0\n",
        encoding="utf-8",
    )
    _, _, altered_bundle = run_pipeline(altered_dir, altered_validation)
    original = solution.load_bundle(original_bundle)
    altered = solution.load_bundle(altered_bundle)
    for original_value, altered_value in zip(original[:3], altered[:3], strict=True):
        np.testing.assert_array_equal(original_value, altered_value)
    assert original[3:] == altered[3:]


def test_validation_metrics_include_known_auc_with_ties_supported() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    result = solution.validation_metrics(y, probabilities, 0.5)
    assert result["accuracy"] == 0.75
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["auc"] == pytest.approx(0.75)


def test_single_test_row_keeps_two_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "one.csv"
    source.write_text("sample_id,feature_1,feature_2\nQ1,1,2\n", encoding="utf-8")
    ids, X, y = solution.load_classification_csv(source, has_label=False)
    assert ids == ["Q1"] and X.shape == (1, 2) and y is None


@pytest.mark.parametrize(
    "content, has_label, message",
    [
        ("sample_id,feature_1,label\nA,1,0\n", True, "表头"),
        ("sample_id,feature_1,feature_2\nA,1,2\nA,2,3\n", False, "重复"),
        ("sample_id,feature_1,feature_2,label\nA,1,2,2\n", True, "必须是0或1"),
        ("sample_id,feature_1,feature_2,label\nA,inf,2,1\n", True, "非有限"),
    ],
)
def test_invalid_classification_csv_is_rejected(
    tmp_path: Path, content: str, has_label: bool, message: str
) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        solution.load_classification_csv(source, has_label=has_label)


def test_auc_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="同时含正负类"):
        solution.validation_metrics(np.ones(3, dtype=int), np.array([0.6, 0.7, 0.8]), 0.5)
