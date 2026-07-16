import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "integrated_tasks" / "02_linear_regression_csv"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("integrated_linear_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_training_data_recovers_known_model() -> None:
    ids, X, y = solution.load_regression_csv(TOPIC / "data" / "train.csv", has_target=True)
    assert len(ids) == 6 and X.shape == (6, 2) and y is not None
    weights, bias = solution.fit_least_squares(X, y)
    np.testing.assert_allclose(weights, [2.0, 3.0], atol=1e-12)
    assert bias == pytest.approx(1.0, abs=1e-12)


def test_model_round_trip_uses_exact_path(tmp_path: Path) -> None:
    path = tmp_path / "model.data"
    solution.save_model(path, np.array([2.0, 3.0]), 1.0)
    weights, bias = solution.load_model(path)
    assert path.is_file() and not (tmp_path / "model.data.npz").exists()
    np.testing.assert_array_equal(weights, [2.0, 3.0])
    assert bias == 1.0


def test_full_command_line_outputs_are_byte_exact(tmp_path: Path) -> None:
    predictions = tmp_path / "out" / "predictions.csv"
    metrics = tmp_path / "out" / "metrics.txt"
    model = tmp_path / "model" / "linear.npz"
    completed = subprocess.run(
        [
            sys.executable, str(SOLUTION),
            "--train", str(TOPIC / "data" / "train.csv"),
            "--validation", str(TOPIC / "data" / "validation.csv"),
            "--test", str(TOPIC / "data" / "test.csv"),
            "--predictions", str(predictions),
            "--metrics", str(metrics),
            "--model", str(model),
        ],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert predictions.read_bytes() == (TOPIC / "expected" / "predictions.csv").read_bytes()
    assert metrics.read_bytes() == (TOPIC / "expected" / "metrics.txt").read_bytes()
    weights, bias = solution.load_model(model)
    np.testing.assert_allclose(weights, [2.0, 3.0], atol=1e-12)
    assert bias == pytest.approx(1.0, abs=1e-12)


def test_validation_targets_do_not_change_fitted_model(tmp_path: Path) -> None:
    altered_validation = tmp_path / "validation.csv"
    altered_validation.write_text(
        "sample_id,feature_1,feature_2,target\nV1,3,0,999\n",
        encoding="utf-8",
    )
    model = tmp_path / "model.npz"
    solution.run_pipeline(
        TOPIC / "data" / "train.csv", altered_validation,
        TOPIC / "data" / "test.csv", tmp_path / "pred.csv",
        tmp_path / "metrics.txt", model,
    )
    weights, bias = solution.load_model(model)
    np.testing.assert_allclose(weights, [2.0, 3.0], atol=1e-12)
    assert bias == pytest.approx(1.0, abs=1e-12)


def test_single_row_test_is_two_dimensional(tmp_path: Path) -> None:
    source = tmp_path / "one.csv"
    source.write_text("sample_id,feature_1,feature_2\nP1,1,2\n", encoding="utf-8")
    _, X, y = solution.load_regression_csv(source, has_target=False)
    assert X.shape == (1, 2) and y is None


@pytest.mark.parametrize(
    "content, has_target, message",
    [
        ("sample_id,feature_1,target\nA,1,2\n", True, "表头"),
        ("sample_id,feature_1,feature_2\nA,1,2\nA,2,3\n", False, "重复"),
        ("sample_id,feature_1,feature_2,target\nA,nan,2,3\n", True, "非有限"),
    ],
)
def test_invalid_csv_is_rejected(tmp_path: Path, content: str, has_target: bool, message: str) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        solution.load_regression_csv(source, has_target=has_target)
