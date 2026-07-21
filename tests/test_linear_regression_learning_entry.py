import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "02_machine_learning" / "01_linear_regression"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_shows_expected_output_and_unfinished_result() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "demo.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "prediction:" in result.stdout
    assert "predict" in result.stdout
    assert "shape" not in result.stdout
    assert "Traceback" not in result.stderr


def test_starter_defers_basic_input_validation() -> None:
    source = (TOPIC / "starter.py").read_text(encoding="utf-8")
    assert "def _check_" not in source
    assert "X.ndim" not in source


def test_simple_checks_accept_reference_implementation(monkeypatch, capsys) -> None:
    monkeypatch.syspath_prepend(str(TOPIC))
    checker = load_module("linear_regression_checker", TOPIC / "check.py")
    solution = load_module(
        "linear_regression_reference_for_checker", TOPIC / "reference" / "solution.py"
    )
    checker.starter = solution

    assert checker.main() == 0
    output = capsys.readouterr().out
    assert "5/5" in output
    assert "True" in output
