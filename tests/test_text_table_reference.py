import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "01_file_io" / "01_text_table"
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


def test_sample_file_writes_expected_result(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "result.txt"
    completed = run_solution(TOPIC / "data" / "sample.txt", output_path)

    expected = (TOPIC / "expected" / "column_means.txt").read_text(
        encoding="utf-8"
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert output_path.read_text(encoding="utf-8") == expected


def test_inconsistent_columns_are_rejected(tmp_path: Path) -> None:
    output_path = tmp_path / "result.txt"
    completed = run_solution(TOPIC / "data" / "bad_columns.txt", output_path)

    assert completed.returncode != 0
    assert "实际得到 2 列" in completed.stderr
    assert not output_path.exists()


def test_missing_file_is_reported(tmp_path: Path) -> None:
    output_path = tmp_path / "result.txt"
    completed = run_solution(tmp_path / "missing.txt", output_path)

    assert completed.returncode != 0
    assert "数据错误" in completed.stderr
    assert not output_path.exists()
