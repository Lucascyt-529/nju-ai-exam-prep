import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "01_file_io" / "04_mixed_csv"
SOLUTION = TOPIC / "reference" / "solution.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("mixed_csv_solution", SOLUTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_solution_module()


def test_load_compute_and_save_strict_output(tmp_path: Path) -> None:
    sample_ids, names, X, y = solution.load_mixed_dataset(TOPIC / "data" / "students.csv")
    means = solution.compute_row_feature_means(X)
    output = tmp_path / "nested" / "summary.csv"
    solution.save_summary(output, sample_ids, names, means, y)

    assert sample_ids == ["S001", "S002", "S003"]
    assert names == ["小林", "小周", "小陈"]
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    np.testing.assert_allclose(means, [5.5, 11.0, 16.5])
    assert output.read_bytes() == (TOPIC / "expected" / "summary.csv").read_bytes()


def test_single_row_stays_two_dimensional(tmp_path: Path) -> None:
    source = tmp_path / "one.csv"
    source.write_text(
        "sample_id,name,feature_1,feature_2,label\nS001,小林,1,2,0\n",
        encoding="utf-8",
    )
    _, _, X, y = solution.load_mixed_dataset(source)
    assert X.shape == (1, 2)
    assert y.shape == (1,)


@pytest.mark.parametrize(
    "content, message",
    [
        ("sample_id,name,feature_1,label\nS1,A,1,0\n", "表头"),
        (
            "sample_id,name,feature_1,feature_2,label\nS1,A,1,2,0\nS1,B,3,4,1\n",
            "重复",
        ),
        ("sample_id,name,feature_1,feature_2,label\nS1,A,bad,2,0\n", "非法"),
        ("sample_id,name,feature_1,feature_2,label\nS1,A,1,2\n", "字段数量"),
    ],
)
def test_bad_mixed_csv_is_rejected(tmp_path: Path, content: str, message: str) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        solution.load_mixed_dataset(source)
