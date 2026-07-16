import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "01_file_io" / "02_clean_csv"
SOLUTION = TOPIC / "reference" / "solution.py"


def run_solution(input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SOLUTION),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def test_clean_csv_writes_exact_expected_output(tmp_path: Path) -> None:
    output_path = tmp_path / "result.csv"

    completed = run_solution(TOPIC / "data" / "train.csv", output_path)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert output_path.read_bytes() == (
        TOPIC / "expected" / "feature_sums.csv"
    ).read_bytes()


def test_clean_csv_single_row_stays_two_dimensional(tmp_path: Path) -> None:
    input_path = tmp_path / "one.csv"
    output_path = tmp_path / "result.csv"
    input_path.write_text(
        "sample_id,feature_1,feature_2,label\n9,1.5,2.5,1\n",
        encoding="utf-8",
    )

    completed = run_solution(input_path, output_path)

    assert completed.returncode == 0
    result = np.loadtxt(output_path, delimiter=",", skiprows=1, ndmin=2)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[9, 4]])


def test_loadtxt_topic_rejects_missing_value(tmp_path: Path) -> None:
    input_path = tmp_path / "missing.csv"
    output_path = tmp_path / "result.csv"
    input_path.write_text(
        "sample_id,feature_1,feature_2,label\n9,1.5,,1\n",
        encoding="utf-8",
    )

    completed = run_solution(input_path, output_path)

    assert completed.returncode != 0
    assert not output_path.exists()
