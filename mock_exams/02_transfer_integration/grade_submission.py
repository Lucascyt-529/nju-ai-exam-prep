"""模拟卷2考后评分器：公开格式与隐藏边界初筛。"""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


EXAM = Path(__file__).resolve().parent
DATA = EXAM / "data"
EXPECTED = EXAM / "expected"


def run_script(script: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def grade_q1(script: Path, temp: Path) -> tuple[int, list[str]]:
    score, notes = 0, []
    output = temp / "q1" / "summary.csv"
    result = run_script(
        script, ["--input", str(DATA / "q1_transactions.csv"), "--output", str(output)]
    )
    if result.returncode == 0 and output.is_file():
        if output.read_bytes() == (EXPECTED / "q1_summary.csv").read_bytes():
            score += 18
        else:
            notes.append("正式汇总数值、排序或格式错误")
    else:
        notes.append("正式样例未运行成功")

    duplicate = temp / "q1_duplicate.csv"
    duplicate.write_text(
        "record_id,category,amount\nR,A,1\nR,B,2\n", encoding="utf-8"
    )
    bad = run_script(script, ["--input", str(duplicate), "--output", str(temp / "d.csv")])
    if bad.returncode != 0:
        score += 3
    else:
        notes.append("未拒绝重复ID")

    negative = temp / "q1_negative.csv"
    negative.write_text("record_id,category,amount\nR,A,-1\n", encoding="utf-8")
    bad = run_script(script, ["--input", str(negative), "--output", str(temp / "n.csv")])
    if bad.returncode != 0:
        score += 2
    else:
        notes.append("未拒绝负金额")

    tie = temp / "q1_tie.csv"
    tie.write_text(
        "record_id,category,amount\n1,B,5\n2,A,5\n", encoding="utf-8"
    )
    tie_output = temp / "tie.csv"
    result = run_script(script, ["--input", str(tie), "--output", str(tie_output)])
    expected = (
        "category,count,total,mean\n"
        "A,1,5.000000,5.000000\n"
        "B,1,5.000000,5.000000\n"
    )
    if result.returncode == 0 and tie_output.is_file() and tie_output.read_text(encoding="utf-8").replace("\r\n", "\n") == expected:
        score += 2
    else:
        notes.append("总额平局时类别顺序错误")
    return min(score, 25), notes


def grade_q2(script: Path, temp: Path) -> tuple[int, list[str]]:
    score, notes = 0, []
    output = temp / "q2" / "path.txt"
    result = run_script(
        script, ["--input", str(DATA / "q2_graph.txt"), "--output", str(output)]
    )
    if result.returncode == 0 and output.is_file() and output.read_bytes() == (EXPECTED / "q2_path.txt").read_bytes():
        score += 30
    else:
        notes.append("正式图的距离、路径或格式错误")

    def check(name: str, content: str, expected: str, points: int) -> None:
        nonlocal score
        source, target = temp / f"{name}.txt", temp / f"{name}_out.txt"
        source.write_text(content, encoding="utf-8")
        completed = run_script(script, ["--input", str(source), "--output", str(target)])
        if completed.returncode == 0 and target.is_file() and target.read_text(encoding="utf-8").replace("\r\n", "\n") == expected:
            score += points
        else:
            notes.append(f"{name}未通过")

    check(
        "等距完整路径",
        "4 4\nS B 1\nB T 1\nS A 1\nA T 1\nS T\n",
        "distance=2.000000\npath=S->A->T\n",
        5,
    )
    check(
        "不可达",
        "4 2\nA B 1\nC D 1\nA D\n",
        "distance=INF\npath=NONE\n",
        5,
    )
    check(
        "起终点相同",
        "2 1\nA B 2\nA A\n",
        "distance=0.000000\npath=A\n",
        3,
    )
    negative = temp / "negative_graph.txt"
    negative.write_text("2 1\nA B -1\nA B\n", encoding="utf-8")
    bad = run_script(script, ["--input", str(negative), "--output", str(temp / "negative.txt")])
    if bad.returncode != 0:
        score += 2
    else:
        notes.append("未拒绝负边权")
    return min(score, 45), notes


def load_npz(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError):
        return None


def execute_q3(script: Path, temp: Path, test_path: Path, suffix: str):
    model = temp / f"q3_{suffix}.npz"
    predictions = temp / f"q3_{suffix}.csv"
    metrics = temp / f"q3_{suffix}.txt"
    result = run_script(
        script,
        [
            "--train", str(DATA / "q3_train.csv"),
            "--validation", str(DATA / "q3_validation.csv"),
            "--test", str(test_path),
            "--model", str(model),
            "--predictions", str(predictions),
            "--metrics", str(metrics),
        ],
    )
    return result, model, predictions, metrics


def grade_q3(script: Path, temp: Path) -> tuple[int, list[str]]:
    score, notes = 0, []
    result, model_path, predictions, metrics = execute_q3(
        script, temp, DATA / "q3_test.csv", "base"
    )
    if result.returncode != 0 or not all(path.is_file() for path in (model_path, predictions, metrics)):
        return 0, ["正式PCA+kNN流程未完整运行"]
    score += 10
    if predictions.read_bytes() == (EXPECTED / "q3_predictions.csv").read_bytes():
        score += 10
    else:
        notes.append("测试预测、顺序或格式错误")
    if metrics.read_bytes() == (EXPECTED / "q3_metrics.txt").read_bytes():
        score += 10
    else:
        notes.append("k选择、验证accuracy或指标格式错误")

    model = load_npz(model_path)
    expected_keys = {"mean", "std", "components", "train_projection", "train_labels", "selected_k"}
    valid = (
        model is not None
        and set(model) == expected_keys
        and model["mean"].shape == (3,)
        and model["std"].shape == (3,)
        and model["components"].shape == (2, 3)
        and model["train_projection"].shape == (12, 2)
        and model["train_labels"].shape == (12,)
        and model["selected_k"].shape == ()
        and all(np.all(np.isfinite(value)) for value in model.values())
        and np.all(model["std"] > 0)
    )
    if valid:
        score += 15
    else:
        notes.append("模型键、形状、安全恢复或有限性错误")
    if model is not None and np.allclose(model["mean"], [0.0, 0.0, -1 / 18]) and np.allclose(model["std"], [2.1015867, 1.55456318, 0.43033148]) and np.array_equal(np.sort(np.unique(model["train_labels"])), [10, 20]):
        score += 10
    else:
        notes.append("最终模型未按训练+验证数据从头拟合")

    changed = temp / "changed_test.csv"
    changed.write_text(
        "sample_id,feature_1,feature_2,feature_3\nZ,999,999,999\n",
        encoding="utf-8",
    )
    changed_result, changed_model_path, _, _ = execute_q3(script, temp, changed, "changed")
    changed_model = load_npz(changed_model_path)
    if changed_result.returncode == 0 and model is not None and changed_model is not None and set(model) == set(changed_model) and all(np.array_equal(model[key], changed_model[key]) for key in model):
        score += 10
    else:
        notes.append("改变测试集后保存模型发生变化")

    single = temp / "single_test.csv"
    single.write_text(
        "sample_id,feature_1,feature_2,feature_3\nONE,-2,-1,\n", encoding="utf-8"
    )
    single_result, _, single_predictions, _ = execute_q3(script, temp, single, "single")
    if single_result.returncode == 0 and single_predictions.is_file():
        try:
            with single_predictions.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            if len(rows) == 1 and rows[0] == {"sample_id": "ONE", "label": "10"}:
                score += 10
            else:
                notes.append("单行测试形状或缺失填补错误")
        except OSError:
            notes.append("单行测试输出无法读取")
    else:
        notes.append("单行测试未运行成功")

    malformed = temp / "bad_test.csv"
    malformed.write_text(
        "sample_id,feature_1,feature_2,feature_3\nBAD,nan,1,2\n", encoding="utf-8"
    )
    bad_result, _, _, _ = execute_q3(script, temp, malformed, "bad")
    if bad_result.returncode != 0:
        score += 5
    else:
        notes.append("未拒绝显式nan")
    return min(score, 80), notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    submission = args.submission.resolve()
    names = ("q1_group_summary.py", "q2_dijkstra.py", "q3_pca_knn.py")
    required = [submission / name for name in names]
    if any(not path.is_file() for path in required):
        print("作答目录必须包含" + "、".join(names), file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="mock_exam_02_") as directory:
        temp = Path(directory)
        results = [grade_q1(required[0], temp), grade_q2(required[1], temp), grade_q3(required[2], temp)]
    maxima = (25, 45, 80)
    total = 0
    for index, ((score, notes), maximum) in enumerate(zip(results, maxima, strict=True), start=1):
        total += score
        print(f"Q{index}={score}/{maximum}")
        for note in notes:
            print(f"  - {note}")
    print(f"TOTAL={total}/150")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
