import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = [
    "10_two_pointers_sliding_window", "11_prefix_sum", "12_stack_monotonic_stack",
    "13_heap_topk", "14_linked_list", "15_binary_tree", "16_backtracking",
    "17_greedy_intervals", "18_topological_sort",
]

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def test_each_pattern_has_complete_learning_surface():
    required={"README.md","starter.py","demo.py","check.py"}
    for topic in TOPICS:
        directory=ROOT/"algorithms"/topic
        assert required.issubset({path.name for path in directory.iterdir()})
        assert (directory/"reference"/"solution.py").is_file()
        assert any((directory/"tests").glob("test_*.py"))
        readme=(directory/"README.md").read_text(encoding="utf-8")
        assert "输入" in readme and "输出" in readme and "O(" in readme and "变式" in readme
        source=(directory/"reference"/"solution.py").read_text(encoding="utf-8")
        assert "def main" in source

def test_all_daily_checks_accept_reference(monkeypatch, capsys):
    for index,topic in enumerate(TOPICS):
        directory=ROOT/"algorithms"/topic
        monkeypatch.syspath_prepend(str(directory))
        checker=load(f"pattern_checker_{index}",directory/"check.py")
        checker.starter=load(f"pattern_reference_{index}",directory/"reference"/"solution.py")
        assert checker.main()==0
    assert capsys.readouterr().out.count("通过:")>=9
