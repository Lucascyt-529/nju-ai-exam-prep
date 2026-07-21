import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MACHINE_LEARNING = ROOT / "02_machine_learning"
TOPICS = [
    "00_model_evaluation",
    "02_logistic_regression",
    "03_knn",
    "04_pca",
    "05_kmeans",
    "06_decision_tree",
    "07_naive_bayes",
    "08_neural_network",
    "09_ensemble_learning/01_adaboost",
    "09_ensemble_learning/02_bagging_oob",
    "09_ensemble_learning/03_random_forest",
    "10_lda",
    "11_svm",
]
ALGORITHM_TOPICS = ["01_linear_regression", *TOPICS[1:]]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative_topic", TOPICS)
def test_learning_entry_has_daily_files(relative_topic: str) -> None:
    topic = MACHINE_LEARNING / relative_topic
    for relative_file in [
        "README.md",
        "starter.py",
        "demo.py",
        "check.py",
        "reference/solution.py",
    ]:
        assert (topic / relative_file).is_file()


@pytest.mark.parametrize("relative_topic", TOPICS)
def test_readme_supports_independent_study(relative_topic: str) -> None:
    readme = (MACHINE_LEARNING / relative_topic / "README.md").read_text(
        encoding="utf-8"
    )
    assert len(readme.splitlines()) >= 70
    assert "## 运行与核对" in readme
    assert "## 常见错误" in readme
    assert "## 自学闭环" in readme


@pytest.mark.parametrize("relative_topic", ALGORITHM_TOPICS)
def test_algorithm_readme_starts_with_an_overview(relative_topic: str) -> None:
    readme = (MACHINE_LEARNING / relative_topic / "README.md").read_text(
        encoding="utf-8"
    )
    assert "## 算法概览" in readme


def test_evaluation_and_ensemble_overviews_explain_the_big_picture() -> None:
    evaluation = (MACHINE_LEARNING / "00_model_evaluation" / "README.md").read_text(
        encoding="utf-8"
    )
    ensemble = (MACHINE_LEARNING / "09_ensemble_learning" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "## 基本认识" in evaluation
    assert "## 算法概览" in ensemble


@pytest.mark.parametrize("relative_topic", TOPICS)
def test_demo_runs_without_traceback(relative_topic: str) -> None:
    topic = MACHINE_LEARNING / relative_topic
    result = subprocess.run(
        [sys.executable, str(topic / "demo.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "期望" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("relative_topic", TOPICS)
def test_simple_check_accepts_reference_implementation(
    relative_topic: str, monkeypatch, capsys
) -> None:
    topic = MACHINE_LEARNING / relative_topic
    monkeypatch.syspath_prepend(str(topic))
    checker_name = "checker_" + relative_topic.replace("/", "_")
    solution_name = "solution_" + relative_topic.replace("/", "_")
    checker = load_module(checker_name, topic / "check.py")
    solution = load_module(solution_name, topic / "reference" / "solution.py")
    checker.starter = solution

    assert checker.main() == 0
    output = capsys.readouterr().out
    assert output.strip()
