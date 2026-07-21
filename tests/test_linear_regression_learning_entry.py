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


def test_demo_shows_shapes_and_unfinished_result() -> None:
    result = subprocess.run(
        [sys.executable, str(TOPIC / "demo.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "prediction" in result.stdout
    assert "predict" in result.stdout
    assert "shapes:" in result.stdout
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
    assert "12/12" in output
    assert "True" in output


def test_demo_accepts_reference_implementation(monkeypatch, capsys) -> None:
    monkeypatch.syspath_prepend(str(TOPIC))
    demo = load_module("linear_regression_demo", TOPIC / "demo.py")
    solution = load_module(
        "linear_regression_reference_for_demo", TOPIC / "reference" / "solution.py"
    )
    demo.starter = solution
    demo.main()
    output = capsys.readouterr().out
    assert "loss_history 前5项" in output
    assert "最小二乘一致 = True" in output


def test_check_rejects_prediction_column_shape(monkeypatch, capsys) -> None:
    monkeypatch.syspath_prepend(str(TOPIC))
    checker = load_module("linear_regression_shape_checker", TOPIC / "check.py")
    solution = load_module(
        "linear_regression_reference_for_shape", TOPIC / "reference" / "solution.py"
    )

    class WrongShape:
        def __getattr__(self, name):
            return getattr(solution, name)

        @staticmethod
        def predict(X, w, b):
            return solution.predict(X, w, b).reshape(-1, 1)

    checker.starter = WrongShape()
    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "2. predict shape" in output
    assert "通过: False" in output


def test_loss_history_contract_is_documented() -> None:
    readme = (TOPIC / "README.md").read_text(encoding="utf-8")
    starter = (TOPIC / "starter.py").read_text(encoding="utf-8")
    assert "n_steps + 1" in readme
    assert "n_steps + 1" in starter
