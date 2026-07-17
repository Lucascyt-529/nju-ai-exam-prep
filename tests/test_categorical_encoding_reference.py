import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "03_data_processing" / "02_categorical_encoding"
spec = importlib.util.spec_from_file_location("categorical_encoding_solution", TOPIC / "reference" / "solution.py")
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


TRAIN = np.array([["红", "小"], ["蓝", "大"], ["红", "大"]])
TEST = np.array([["绿", "小"], ["红", "中"]])


def test_vocabularies_are_per_feature_unique_and_sorted() -> None:
    vocabularies = solution.fit_category_vocabularies(TRAIN)
    assert vocabularies == (("红", "蓝"), ("大", "小"))
    assert isinstance(vocabularies, tuple) and all(isinstance(item, tuple) for item in vocabularies)


def test_test_categories_do_not_enter_training_vocabulary() -> None:
    vocabularies = solution.fit_category_vocabularies(TRAIN)
    solution.transform_one_hot(TEST, vocabularies)
    assert "绿" not in vocabularies[0] and "中" not in vocabularies[1]


def test_ordinal_known_categories_and_unknown_policy() -> None:
    vocabularies = solution.fit_category_vocabularies(TRAIN)
    np.testing.assert_array_equal(
        solution.transform_ordinal(TRAIN, vocabularies),
        [[0, 1], [1, 0], [0, 0]],
    )
    with pytest.raises(ValueError, match="未见类别"):
        solution.transform_ordinal(TEST, vocabularies)
    np.testing.assert_array_equal(
        solution.transform_ordinal(TEST, vocabularies, handle_unknown="use_encoded_value", unknown_value=-7),
        [[-7, 1], [0, -7]],
    )


def test_one_hot_blocks_have_expected_shape_and_unknown_last() -> None:
    vocabularies = solution.fit_category_vocabularies(TRAIN)
    encoded = solution.transform_one_hot(TEST, vocabularies, include_unknown=True)
    assert encoded.shape == (2, 6)
    np.testing.assert_array_equal(encoded, [
        [0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
    ])
    np.testing.assert_array_equal(encoded[:, :3].sum(axis=1), 1)
    np.testing.assert_array_equal(encoded[:, 3:].sum(axis=1), 1)


def test_unknown_without_bucket_is_rejected() -> None:
    vocabularies = solution.fit_category_vocabularies(TRAIN)
    with pytest.raises(ValueError, match="未见类别"):
        solution.transform_one_hot(TEST, vocabularies, include_unknown=False)


def test_feature_names_match_every_output_column() -> None:
    vocabularies = solution.fit_category_vocabularies(TRAIN)
    names = solution.one_hot_feature_names(("颜色", "尺寸"), vocabularies)
    assert names == (
        "颜色=红", "颜色=蓝", "颜色=<UNKNOWN>",
        "尺寸=大", "尺寸=小", "尺寸=<UNKNOWN>",
    )
    assert len(names) == solution.transform_one_hot(TRAIN, vocabularies).shape[1]


def test_numeric_and_one_hot_concatenation_preserves_rows() -> None:
    numeric = np.array([[1., 10.], [2., 20.]])
    one_hot = np.array([[1., 0., 0.], [0., 1., 0.]])
    combined = solution.combine_numeric_and_categorical(numeric, one_hot)
    assert combined.shape == (2, 5)
    np.testing.assert_array_equal(combined[:, :2], numeric)
    np.testing.assert_array_equal(combined[:, 2:], one_hot)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([["红", ""]]),
        np.array([["红", 1]], dtype=object),
        np.array(["红", "蓝"]),
        np.empty((0, 2), dtype=str),
    ],
)
def test_invalid_categorical_matrices_are_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        solution.fit_category_vocabularies(bad)


def test_invalid_vocabularies_are_rejected() -> None:
    with pytest.raises(ValueError):
        solution.transform_one_hot(TRAIN, (("蓝", "红"), ("大", "小")))
    with pytest.raises(ValueError):
        solution.transform_one_hot(TRAIN, (("红", "蓝"),))


def test_combination_rejects_mismatched_sample_order_length() -> None:
    with pytest.raises(ValueError, match="样本数"):
        solution.combine_numeric_and_categorical(np.ones((2, 1)), np.ones((3, 2)))


def test_encoder_metadata_json_round_trip_preserves_column_semantics(tmp_path: Path) -> None:
    vocabularies = solution.fit_category_vocabularies(TRAIN)
    path = tmp_path / "nested" / "encoder.json"
    solution.save_encoder_metadata(path, ("颜色", "尺寸"), vocabularies)
    restored = solution.load_encoder_metadata(path)
    assert restored == {
        "feature_names": ("颜色", "尺寸"),
        "vocabularies": vocabularies,
        "include_unknown": True,
    }
    text = path.read_text(encoding="utf-8")
    assert "颜色" in text and "\\u989c" not in text and text.endswith("\n")
    names = solution.one_hot_feature_names(
        restored["feature_names"], restored["vocabularies"],
        include_unknown=restored["include_unknown"],
    )
    assert len(names) == 6


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(format_version=2),
        lambda payload: payload.update(extra="unexpected"),
        lambda payload: payload.update(include_unknown=1),
        lambda payload: payload.update(vocabularies=[["红", "红"], ["大", "小"]]),
    ],
)
def test_corrupt_encoder_metadata_is_rejected(tmp_path: Path, mutation) -> None:
    payload = {
        "format_version": 1,
        "feature_names": ["颜色", "尺寸"],
        "vocabularies": [["红", "蓝"], ["大", "小"]],
        "include_unknown": True,
    }
    mutation(payload)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        solution.load_encoder_metadata(path)


def test_guided_demo_runs_and_makes_leakage_boundary_visible() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "one-hot形状: (2, 6)" in completed.stdout
    assert "测试新类别没有进入训练词表: True" in completed.stdout


def test_starter_remains_for_student() -> None:
    starter_spec = importlib.util.spec_from_file_location("categorical_starter", TOPIC / "starter.py")
    assert starter_spec is not None and starter_spec.loader is not None
    starter = importlib.util.module_from_spec(starter_spec); starter_spec.loader.exec_module(starter)
    with pytest.raises(NotImplementedError):
        starter.fit_category_vocabularies(TRAIN)
