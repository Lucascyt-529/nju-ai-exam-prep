from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TOPICS=["01_linear_regression","02_logistic_regression","03_knn","04_pca","05_kmeans"]
FORBIDDEN=("def ","starter.py","check.py","demo.py","reference/","实现顺序","函数名")

def test_exam_modes_are_data_only_and_do_not_leak_scaffolding():
    for topic in TOPICS:
        directory=ROOT/"02_machine_learning"/topic/"exercises"/"exam_mode"
        assert (directory/"PAPER.md").is_file() and (directory/"README.md").is_file()
        assert (directory/"data").is_dir() and any((directory/"data").iterdir())
        assert not list(directory.rglob("*.py"))
        paper=(directory/"PAPER.md").read_text(encoding="utf-8")
        assert all(word not in paper for word in FORBIDDEN)
        assert "评分" in paper and "程序" in paper and "输出" in paper
