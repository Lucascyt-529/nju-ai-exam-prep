"""考后评分器：对指定作答目录运行公开格式与隐藏边界检查。"""

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


def run_script(script: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False, capture_output=True, text=True, encoding="utf-8", timeout=30,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def grade_q1(script: Path, temp: Path) -> tuple[int, list[str]]:
    score = 0; notes = []
    output = temp / "q1" / "summary.csv"
    result = run_script(script, ["--input", str(DATA / "q1_features.csv"), "--output", str(output)])
    if result.returncode == 0 and output.is_file():
        try:
            with output.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            valid_header = list(rows[0]) == ["feature", "mean", "std", "missing_count"]
            means = np.array([float(row["mean"]) for row in rows])
            stds = np.array([float(row["std"]) for row in rows])
            missing = np.array([int(row["missing_count"]) for row in rows])
            if valid_header and [row["feature"] for row in rows] == ["feature_1", "feature_2", "feature_3"]:
                score += 6
            if np.allclose(means, [3., 27.5, 8.], atol=5e-7): score += 8
            if np.allclose(stds, [np.sqrt(2), np.sqrt(175), 2.], atol=5e-6): score += 6
            if np.array_equal(missing, [1, 1, 1]): score += 4
        except (OSError, ValueError, KeyError, IndexError):
            notes.append("正式输出无法解析")
    else:
        notes.append("正式样例未运行成功")

    duplicate = temp / "q1_duplicate.csv"
    duplicate.write_text("sample_id,feature_1,feature_2,feature_3\nA,1,2,3\nA,4,5,6\n", encoding="utf-8")
    bad = run_script(script, ["--input", str(duplicate), "--output", str(temp / "bad1.csv")])
    if bad.returncode != 0: score += 3
    else: notes.append("未拒绝重复ID")
    all_missing = temp / "q1_missing.csv"
    all_missing.write_text("sample_id,feature_1,feature_2,feature_3\nA,1,,3\nB,2,,4\n", encoding="utf-8")
    bad = run_script(script, ["--input", str(all_missing), "--output", str(temp / "bad2.csv")])
    if bad.returncode != 0: score += 3
    else: notes.append("未拒绝整列缺失")
    return min(score, 30), notes


def grade_q2(script: Path, temp: Path) -> tuple[int, list[str]]:
    score = 0; notes = []

    def check(name: str, content: str, expected: str, points: int) -> None:
        nonlocal score
        source = temp / f"{name}.txt"; output = temp / f"{name}_out.txt"
        source.write_text(content, encoding="utf-8")
        result = run_script(script, ["--input", str(source), "--output", str(output)])
        if result.returncode == 0 and output.is_file() and output.read_text(encoding="utf-8").replace("\r\n", "\n") == expected:
            score += points
        else:
            notes.append(f"{name}未通过")

    sample_output = temp / "q2_sample.txt"
    sample = run_script(script, ["--input", str(DATA / "q2_grid.txt"), "--output", str(sample_output)])
    expected = (EXAM / "expected" / "q2_path.txt").read_bytes()
    if sample.returncode == 0 and sample_output.is_file() and sample_output.read_bytes() == expected:
        score += 30
    else:
        notes.append("正式样例的距离、路径或格式错误")
    check("方向平局", "2 2\n..\n..\n1 1\n0 0\n", "distance=2\npath=(1,1)->(0,1)->(0,0)\n", 5)
    check("不可达", "2 2\n.#\n#.\n0 0\n1 1\n", "distance=-1\npath=NONE\n", 5)
    check("起终点相同", "1 1\n.\n0 0\n0 0\n", "distance=0\npath=(0,0)\n", 5)
    return min(score, 45), notes


def _load_npz(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError):
        return None


def grade_q3(script: Path, temp: Path) -> tuple[int, list[str]]:
    score = 0; notes = []

    def execute(test_path: Path, validation_path: Path, suffix: str):
        model = temp / f"q3_{suffix}.npz"; predictions = temp / f"q3_{suffix}.csv"; metrics = temp / f"q3_{suffix}.txt"
        result = run_script(script, [
            "--train", str(DATA / "q3_train.csv"), "--validation", str(validation_path),
            "--test", str(test_path), "--model", str(model),
            "--predictions", str(predictions), "--metrics", str(metrics),
        ])
        return result, model, predictions, metrics

    result, model_path, predictions_path, metrics_path = execute(
        DATA / "q3_test.csv", DATA / "q3_validation.csv", "base"
    )
    if result.returncode == 0 and all(path.is_file() for path in (model_path, predictions_path, metrics_path)):
        score += 10
    else:
        notes.append("正式流程未完整运行")
        return score, notes
    try:
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        probability = np.array([float(row["probability"]) for row in rows])
        labels = np.array([int(row["label"]) for row in rows])
        if (list(rows[0]) == ["sample_id", "probability", "label"]
                and [row["sample_id"] for row in rows] == ["T01", "T02", "T03", "T04"]):
            score += 5
        if np.all(np.isfinite(probability)) and np.all((0 <= probability) & (probability <= 1)):
            score += 5
        if np.allclose(probability, [0.002499, 0.326962, 0.673038, 0.997501], atol=2e-3):
            score += 5
        if np.array_equal(labels, [0, 0, 1, 1]): score += 5
    except (OSError, ValueError, KeyError, IndexError):
        notes.append("预测文件无法解析")
    normalized_metrics = metrics_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if normalized_metrics == "accuracy=1.000000\nf1=1.000000\n": score += 10
    else: notes.append("验证指标或格式错误")

    base_model = _load_npz(model_path)
    if (base_model is not None and set(base_model) == {"mean", "std", "weights", "bias"}
            and base_model["mean"].shape == (3,) and base_model["std"].shape == (3,)
            and base_model["weights"].shape == (3,) and base_model["bias"].shape == ()
            and all(np.all(np.isfinite(value)) for value in base_model.values())
            and np.all(base_model["std"] > 0)):
        score += 10
    else:
        notes.append("模型键、形状、安全恢复或有限性错误")

    changed_test = temp / "q3_changed_test.csv"
    changed_test.write_text("sample_id,feature_1,feature_2,feature_3\nZ,999,999,999\n", encoding="utf-8")
    changed_result, changed_model_path, _, _ = execute(changed_test, DATA / "q3_validation.csv", "changed_test")
    changed_model = _load_npz(changed_model_path)
    if (changed_result.returncode == 0 and base_model is not None and changed_model is not None
            and set(base_model) == set(changed_model)
            and all(np.array_equal(base_model[key], changed_model[key]) for key in base_model)):
        score += 10
    else:
        notes.append("测试集改变了模型")

    changed_validation = temp / "q3_changed_validation.csv"
    changed_validation.write_text(
        "sample_id,feature_1,feature_2,feature_3,label\nV01,-1.2,-10,0,1\nV02,-0.3,-4,1,1\nV03,0.3,4,-1,0\nV04,1.2,10,0,0\n",
        encoding="utf-8",
    )
    validation_result, validation_model_path, _, _ = execute(DATA / "q3_test.csv", changed_validation, "changed_validation")
    validation_model = _load_npz(validation_model_path)
    if (validation_result.returncode == 0 and base_model is not None and validation_model is not None
            and set(base_model) == set(validation_model)
            and all(np.array_equal(base_model[key], validation_model[key]) for key in base_model)):
        score += 10
    else:
        notes.append("验证标签改变了训练模型")

    malformed = temp / "q3_bad.csv"
    malformed.write_text("sample_id,feature_1,feature_2,feature_3\nA,nan,1,2\n", encoding="utf-8")
    bad_result, _, _, _ = execute(malformed, DATA / "q3_validation.csv", "bad")
    if bad_result.returncode != 0: score += 5
    else: notes.append("未拒绝非法测试数值")
    return min(score, 75), notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args(); submission = args.submission.resolve()
    required = [submission / name for name in ("q1_csv_stats.py", "q2_grid_bfs.py", "q3_logistic.py")]
    if any(not path.is_file() for path in required):
        print("作答目录必须包含q1_csv_stats.py、q2_grid_bfs.py、q3_logistic.py", file=sys.stderr); return 2
    with tempfile.TemporaryDirectory(prefix="mock_exam_01_") as directory:
        temp = Path(directory)
        results = [grade_q1(required[0], temp), grade_q2(required[1], temp), grade_q3(required[2], temp)]
    total = sum(score for score, _ in results)
    for index, (score, notes) in enumerate(results, start=1):
        maximum = (30, 45, 75)[index - 1]
        print(f"Q{index}={score}/{maximum}")
        for note in notes:
            print(f"  - {note}")
    print(f"TOTAL={total}/150")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
