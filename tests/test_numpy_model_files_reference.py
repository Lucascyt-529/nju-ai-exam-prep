import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "01_file_io" / "05_numpy_model_files"
spec = importlib.util.spec_from_file_location("numpy_model_files_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def test_single_array_round_trip_preserves_exact_path_dtype_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "features.data"
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    solution.save_array(path, array)
    restored = solution.load_array(path, expected_ndim=2)
    assert path.is_file() and not (path.parent / "features.data.npy").exists()
    np.testing.assert_array_equal(restored, array)
    assert restored.dtype == array.dtype and restored is not array


def test_array_loader_rejects_wrong_dimension(tmp_path: Path) -> None:
    path = tmp_path / "vector.npy"; solution.save_array(path, np.array([1., 2.]))
    with pytest.raises(ValueError, match="维数"):
        solution.load_array(path, expected_ndim=2)


def test_model_bundle_round_trip_and_exact_key_set(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    mean = np.array([10., 100.]); scale = np.array([2., 20.]); weights = np.array([1.5, -0.5])
    solution.save_model_bundle(path, mean, scale, weights, 0.25)
    bundle = solution.load_model_bundle(path)
    assert set(bundle) == {"mean", "scale", "weights", "bias"}
    assert path.is_file() and not (tmp_path / "model.bin.npz").exists()
    np.testing.assert_array_equal(bundle["mean"], mean)
    np.testing.assert_array_equal(bundle["scale"], scale)
    np.testing.assert_array_equal(bundle["weights"], weights)
    assert bundle["bias"] == 0.25


def test_prediction_uses_restored_preprocessing_parameters() -> None:
    X = np.array([[12., 80.], [8., 140.]])
    bundle = {
        "mean": np.array([10., 100.]), "scale": np.array([2., 20.]),
        "weights": np.array([1.5, -0.5]), "bias": 0.25,
    }
    expected = ((X - bundle["mean"]) / bundle["scale"]) @ bundle["weights"] + 0.25
    np.testing.assert_allclose(solution.predict_with_bundle(X, bundle), expected)


@pytest.mark.parametrize(
    ("mean", "scale", "weights", "bias", "message"),
    [
        (np.ones(2), np.ones(3), np.ones(2), 0.0, "相同形状"),
        (np.ones(2), np.array([1., 0.]), np.ones(2), 0.0, "大于0"),
        (np.ones(2), np.ones(2), np.array([1., np.nan]), 0.0, "有限"),
        (np.ones(2), np.ones(2), np.ones(2), "0", "标量"),
    ],
)
def test_invalid_model_parameters_are_rejected(tmp_path: Path, mean, scale, weights, bias, message) -> None:
    with pytest.raises(ValueError, match=message):
        solution.save_model_bundle(tmp_path / "bad.npz", mean, scale, weights, bias)


def test_loader_rejects_missing_extra_and_wrong_bias_shape(tmp_path: Path) -> None:
    cases = [
        {"mean": np.ones(2), "scale": np.ones(2), "weights": np.ones(2)},
        {"mean": np.ones(2), "scale": np.ones(2), "weights": np.ones(2), "bias": np.array(0.), "extra": np.ones(1)},
        {"mean": np.ones(2), "scale": np.ones(2), "weights": np.ones(2), "bias": np.array([0.])},
    ]
    for index, values in enumerate(cases):
        path = tmp_path / f"bad{index}.npz"
        with path.open("wb") as file:
            np.savez(file, **values)
        with pytest.raises(ValueError):
            solution.load_model_bundle(path)


def test_object_array_is_not_loaded_with_pickle(tmp_path: Path) -> None:
    path = tmp_path / "object.npy"
    with path.open("wb") as file:
        np.save(file, np.array([{"unsafe": True}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        solution.load_array(path)


def test_guided_demo_runs_and_exposes_shapes() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "实际文件名: model.data" in completed.stdout
    assert "mean shape: (2,)" in completed.stdout
    assert "predictions:" in completed.stdout


def test_starter_remains_for_student() -> None:
    starter_spec = importlib.util.spec_from_file_location("numpy_model_files_starter", TOPIC / "starter.py")
    assert starter_spec is not None and starter_spec.loader is not None
    starter = importlib.util.module_from_spec(starter_spec); starter_spec.loader.exec_module(starter)
    with pytest.raises(NotImplementedError):
        starter.save_array(Path("x"), np.ones(1))
