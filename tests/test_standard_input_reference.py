import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "00_python_programming" / "01_standard_input"
SOLUTION = TOPIC / "reference" / "solution.py"


def run_solution(input_text: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SOLUTION)],
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def test_sample_output_matches_expected_file() -> None:
    input_text = (TOPIC / "sample_input.txt").read_text(encoding="utf-8")
    expected = (TOPIC / "expected_output.txt").read_text(encoding="utf-8")

    completed = run_solution(input_text)

    assert completed.returncode == 0
    assert completed.stdout == expected
    assert completed.stderr == ""


def test_wrong_column_count_returns_nonzero_status() -> None:
    completed = run_solution("2 3\n1 2 3\n4 5\n")

    assert completed.returncode != 0
    assert "应有 3 列" in completed.stderr


def test_non_finite_value_is_rejected() -> None:
    completed = run_solution("1 2\n1 nan\n")

    assert completed.returncode != 0
    assert "NaN 或无穷大" in completed.stderr
