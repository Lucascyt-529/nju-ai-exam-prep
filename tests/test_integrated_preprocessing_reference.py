import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "integrated_tasks" / "01_preprocessing_pipeline"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("integrated_preprocessing_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_fit_transform_uses_training_statistics_only() -> None:
    _, X_train, y = solution.load_feature_csv(TOPIC / "data" / "train.csv", has_label=True)
    _, X_test, no_label = solution.load_feature_csv(TOPIC / "data" / "test.csv", has_label=False)
    fill_values, means, scales = solution.fit_preprocessor(X_train)
    transformed = solution.transform_features(X_test, fill_values, means, scales)

    assert y is not None and y.shape == (3,)
    assert no_label is None
    np.testing.assert_allclose(fill_values, [3.0, 20.0])
    np.testing.assert_allclose(means, [3.0, 20.0])
    np.testing.assert_allclose(scales, [np.sqrt(8 / 3), np.sqrt(200 / 3)])
    np.testing.assert_allclose(
        transformed,
        [[0.0, 0.0], [0.0, np.sqrt(6)], [np.sqrt(6), 0.0]],
    )


def test_parameter_round_trip_and_exact_path(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "parameters.data"
    expected = (
        np.array([3.0, 20.0]),
        np.array([3.0, 20.0]),
        np.array([1.5, 8.0]),
    )
    solution.save_parameters(path, *expected)
    actual = solution.load_parameters(path)

    assert path.is_file()
    assert not (tmp_path / "nested" / "parameters.data.npz").exists()
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_full_command_line_pipeline_and_strict_output(tmp_path: Path) -> None:
    output = tmp_path / "deep" / "transformed.csv"
    params = tmp_path / "params" / "preprocessor.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(SOLUTION),
            "--train",
            str(TOPIC / "data" / "train.csv"),
            "--test",
            str(TOPIC / "data" / "test.csv"),
            "--output",
            str(output),
            "--params",
            str(params),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == (
        TOPIC / "expected" / "transformed_test.csv"
    ).read_bytes()
    fill_values, means, scales = solution.load_parameters(params)
    np.testing.assert_allclose(fill_values, [3.0, 20.0])
    np.testing.assert_allclose(means, [3.0, 20.0])
    np.testing.assert_allclose(scales, [np.sqrt(8 / 3), np.sqrt(200 / 3)])


def test_single_test_row_keeps_two_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "one.csv"
    source.write_text(
        "sample_id,feature_1,feature_2\nT1,1,2\n", encoding="utf-8"
    )
    ids, X, y = solution.load_feature_csv(source, has_label=False)
    assert ids == ["T1"]
    assert X.shape == (1, 2)
    assert y is None


def test_all_missing_training_column_is_rejected() -> None:
    X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    with pytest.raises(ValueError, match="整列缺失"):
        solution.fit_preprocessor(X)


@pytest.mark.parametrize(
    "content, has_label, message",
    [
        ("sample_id,feature_1,label\nA,1,0\n", True, "表头"),
        (
            "sample_id,feature_1,feature_2\nT1,1,2\nT1,3,4\n",
            False,
            "重复",
        ),
        ("sample_id,feature_1,feature_2,label\nA,1,2,0.5\n", True, "label"),
        ("sample_id,feature_1,feature_2\nT1,inf,2\n", False, "不能是"),
    ],
)
def test_invalid_csv_is_rejected(
    tmp_path: Path, content: str, has_label: bool, message: str
) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        solution.load_feature_csv(source, has_label=has_label)


def test_corrupted_parameter_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, means=np.array([0.0]))
    with pytest.raises(ValueError, match="必须包含"):
        solution.load_parameters(path)
