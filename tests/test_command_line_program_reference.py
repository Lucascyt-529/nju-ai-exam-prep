import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "00_python_programming" / "02_command_line_program"
SOLUTION = TOPIC / "reference" / "solution.py"
spec = importlib.util.spec_from_file_location("command_line_program_solution", SOLUTION)
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)


def test_importing_module_does_not_run_program(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    module_spec = importlib.util.spec_from_file_location("command_line_import_again", SOLUTION)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    assert list(tmp_path.iterdir()) == []


def test_loader_skips_blank_lines_and_returns_vector() -> None:
    values = solution.load_values(TOPIC / "data" / "values.txt")
    assert values.shape == (4,) and values.dtype == float
    np.testing.assert_array_equal(values, [1., 2., 3., 4.])


def test_statistics_match_hand_values_and_do_not_modify_input() -> None:
    values = np.array([1., 2., 3., 4.]); before = values.copy()
    report = solution.compute_statistics(values, scale=2)
    assert report == {
        "count": 4, "mean": 2.5, "std": pytest.approx(np.sqrt(1.25)),
        "minimum": 1.0, "maximum": 4.0, "scaled_mean": 5.0,
    }
    np.testing.assert_array_equal(values, before)


def test_run_pipeline_creates_parent_and_byte_exact_report(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.txt"
    solution.run_pipeline(TOPIC / "data" / "values.txt", output, scale=2)
    assert output.read_bytes() == (TOPIC / "expected" / "report.txt").read_bytes()


def test_main_accepts_explicit_argv_without_touching_global_argv(tmp_path: Path) -> None:
    output = tmp_path / "report.txt"; before = sys.argv.copy()
    code = solution.main([
        "--input", str(TOPIC / "data" / "values.txt"),
        "--output", str(output), "--scale", "2",
    ])
    assert code == 0 and output.is_file() and sys.argv == before


def test_real_process_success_and_exit_code(tmp_path: Path) -> None:
    output = tmp_path / "report.txt"
    completed = subprocess.run(
        [sys.executable, str(SOLUTION), "--input", str(TOPIC / "data" / "values.txt"),
         "--output", str(output), "--scale", "2"],
        check=False, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert completed.returncode == 0 and completed.stderr == ""
    assert output.read_bytes() == (TOPIC / "expected" / "report.txt").read_bytes()


def test_expected_input_error_returns_two_and_stderr(tmp_path: Path, capsys) -> None:
    code = solution.main([
        "--input", str(tmp_path / "missing.txt"),
        "--output", str(tmp_path / "report.txt"),
    ])
    captured = capsys.readouterr()
    assert code == 2 and "程序失败" in captured.err
    assert not (tmp_path / "report.txt").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [("1 2\n", "恰好"), ("abc\n", "合法"), ("nan\n", "有限"), ("\n  \n", "没有有效")],
)
def test_malformed_input_is_rejected_with_specific_reason(tmp_path: Path, content: str, message: str) -> None:
    source = tmp_path / "bad.txt"; source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        solution.load_values(source)


def test_parser_has_default_scale() -> None:
    args = solution.build_parser().parse_args(["--input", "in.txt", "--output", "out.txt"])
    assert args.input == Path("in.txt") and args.output == Path("out.txt") and args.scale == 1.0


def test_guided_demo_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOPIC / "guided_demo.py")], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert "导入模块时没有自动执行命令行程序" in completed.stdout
    assert "main返回值: 0" in completed.stdout
    assert "scaled_mean=5.000000" in completed.stdout


def test_starter_remains_for_student() -> None:
    starter_spec = importlib.util.spec_from_file_location("command_line_starter", TOPIC / "starter.py")
    assert starter_spec is not None and starter_spec.loader is not None
    starter = importlib.util.module_from_spec(starter_spec); starter_spec.loader.exec_module(starter)
    with pytest.raises(NotImplementedError):
        starter.load_values(Path("input.txt"))
