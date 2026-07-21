import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "02_machine_learning" / "00_model_evaluation"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def test_daily_checks_accept_reference(monkeypatch, capsys):
    for topic in ("classification", "data_splitting"):
        directory = BASE / topic
        monkeypatch.syspath_prepend(str(directory))
        checker = load(f"{topic}_checker", directory / "check.py")
        checker.starter = load(f"{topic}_solution", directory / "reference" / "solution.py")
        assert checker.main() == 0
    output = capsys.readouterr().out
    assert "通过: 8/8" in output
    assert "通过: 4/4" in output

def test_subtopics_have_complete_learning_files():
    required = {"README.md", "starter.py", "demo.py", "check.py"}
    for topic in ("classification", "data_splitting"):
        directory = BASE / topic
        assert required.issubset({path.name for path in directory.iterdir()})
        assert (directory / "reference" / "solution.py").is_file()
        assert any((directory / "tests").glob("test_*.py"))
