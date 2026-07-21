import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; TOPIC=ROOT/"integrated_tasks"/"02_linear_regression_csv"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def test_demo_and_check_accept_reference(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(TOPIC))
    solution=load("integrated_lr_entry_reference",TOPIC/"reference"/"solution.py")
    demo=load("integrated_lr_demo",TOPIC/"demo.py"); demo.starter=solution; demo.main()
    checker=load("integrated_lr_check",TOPIC/"check.py"); checker.starter=solution; assert checker.main()==0
    output=capsys.readouterr().out; assert "前两行" in output and "通过: 10/10" in output
def test_exam_mode_has_no_python_or_scaffolding():
    directory=TOPIC/"exam_mode"; assert not list(directory.rglob("*.py"))
    paper=(directory/"PAPER.md").read_text(encoding="utf-8")
    for forbidden in ("def ","starter","reference","实现顺序","函数名"): assert forbidden not in paper
