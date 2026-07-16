import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "01_file_io" / "03_missing_csv"
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


def test_missing_csv_writes_exact_expected_output(tmp_path: Path) -> None:
    output_path = tmp_path / "cleaned.csv"

    completed = run_solution(TOPIC / "data" / "train_missing.csv", output_path)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert output_path.read_bytes() == (
        TOPIC / "expected" / "cleaned.csv"
    ).read_bytes()


def test_missing_csv_uses_feature_mean(tmp_path: Path) -> None:
    output_path = tmp_path / "cleaned.csv"
    completed = run_solution(TOPIC / "data" / "train_missing.csv", output_path)

    data = np.loadtxt(output_path, delimiter=",", skiprows=1, ndmin=2)

    assert completed.returncode == 0
    assert data[0, 2] == 25.0


def test_missing_csv_rejects_entirely_missing_feature(tmp_path: Path) -> None:
    input_path = tmp_path / "all_missing.csv"
    output_path = tmp_path / "cleaned.csv"
    input_path.write_text(
        "sample_id,feature_1,feature_2,label\n1,1.0,,0\n2,2.0,,1\n",
        encoding="utf-8",
    )

    completed = run_solution(input_path, output_path)

    assert completed.returncode != 0
    assert "整列缺失" in completed.stderr
    assert not output_path.exists()
