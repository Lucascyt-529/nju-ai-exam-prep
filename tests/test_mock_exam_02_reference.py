import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAM = ROOT / "mock_exams" / "02_transfer_integration"


def load_module(name: str):
    path = EXAM / "reference" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mock2_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q1 = load_module("q1_group_summary")
q2 = load_module("q2_dijkstra")
q3 = load_module("q3_pca_knn")


def test_q1_group_summary_values_and_order() -> None:
    rows = q1.load_rows(EXAM / "data" / "q1_transactions.csv")
    assert q1.summarize(rows) == pytest.approx(
        [("A", 3, 35.0, 35 / 3), ("B", 2, 20.0, 10.0), ("C", 1, 8.0, 8.0)]
    )


def test_q1_output_is_byte_exact(tmp_path: Path) -> None:
    output = tmp_path / "new" / "summary.csv"
    q1.run(EXAM / "data" / "q1_transactions.csv", output)
    assert output.read_bytes() == (EXAM / "expected" / "q1_summary.csv").read_bytes()


def test_q1_tie_uses_category_and_rejects_duplicate(tmp_path: Path) -> None:
    assert [row[0] for row in q1.summarize([("1", "B", 5.0), ("2", "A", 5.0)])] == ["A", "B"]
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("record_id,category,amount\nR,A,1\nR,B,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        q1.load_rows(duplicate)


def test_q1_rejects_negative_and_nonfinite_amount(tmp_path: Path) -> None:
    for token in ("-1", "nan", "inf"):
        source = tmp_path / f"bad_{token}.csv"
        source.write_text(f"record_id,category,amount\nR,A,{token}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            q1.load_rows(source)


def test_q2_sample_distance_path_and_exact_output(tmp_path: Path) -> None:
    graph, start, goal = q2.load_problem(EXAM / "data" / "q2_graph.txt")
    assert q2.shortest_path(graph, start, goal) == (13.0, ["A", "C", "B", "D", "E", "F"])
    output = tmp_path / "new" / "path.txt"
    q2.run(EXAM / "data" / "q2_graph.txt", output)
    assert output.read_bytes() == (EXAM / "expected" / "q2_path.txt").read_bytes()


def test_q2_equal_distance_uses_full_path_lexicographic_order() -> None:
    graph = {
        "S": [("B", 1.0), ("A", 1.0)],
        "A": [("S", 1.0), ("T", 1.0)],
        "B": [("S", 1.0), ("T", 1.0)],
        "T": [("A", 1.0), ("B", 1.0)],
    }
    assert q2.shortest_path(graph, "S", "T") == (2.0, ["S", "A", "T"])


def test_q2_start_goal_and_unreachable() -> None:
    graph = {"A": [("B", 1.0)], "B": [("A", 1.0)], "C": []}
    assert q2.shortest_path(graph, "A", "A") == (0.0, ["A"])
    assert q2.shortest_path(graph, "A", "C") is None


def test_q2_parser_rejects_negative_duplicate_and_self_loop(tmp_path: Path) -> None:
    cases = [
        "2 1\nA B -1\nA B\n",
        "2 2\nA B 1\nB A 2\nA B\n",
        "1 1\nA A 1\nA A\n",
    ]
    for index, content in enumerate(cases):
        source = tmp_path / f"bad_graph_{index}.txt"
        source.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            q2.load_problem(source)


def test_q3_csv_shapes_missing_and_noncontinuous_labels() -> None:
    ids, X, y = q3.load_csv(EXAM / "data" / "q3_train.csv", has_label=True)
    assert len(ids) == 8 and X.shape == (8, 3) and y is not None and y.shape == (8,)
    assert np.isnan(X).sum() == 2
    np.testing.assert_array_equal(np.unique(y), [10, 20])


def test_q3_preprocessor_uses_current_fit_data_and_safe_std() -> None:
    X = np.array([[1.0, np.nan, 5.0], [3.0, 4.0, 5.0]])
    mean, std = q3.fit_preprocessor(X)
    np.testing.assert_allclose(mean, [2.0, 4.0, 5.0])
    np.testing.assert_allclose(std, [1.0, 1.0, 1.0])
    scaled = q3.transform(X, mean, std)
    np.testing.assert_allclose(scaled, [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_q3_pca_components_are_ordered_orthonormal_and_sign_fixed() -> None:
    _, X, y = q3.load_csv(EXAM / "data" / "q3_train.csv", has_label=True)
    assert y is not None
    mean, std = q3.fit_preprocessor(X)
    components, eigenvalues = q3.fit_pca(q3.transform(X, mean, std), 2)
    assert components.shape == (2, 3)
    assert eigenvalues[0] >= eigenvalues[1] >= -1e-12
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-12)
    for row in components:
        assert row[np.argmax(np.abs(row))] >= 0


def test_q3_knn_vote_distance_and_label_ties_are_deterministic() -> None:
    train = np.array([[0.0], [2.0], [4.0], [6.0]])
    labels = np.array([20, 10, 20, 10])
    query = np.array([[3.0]])
    assert q3.knn_predict(train, labels, query, 2)[0] == 10
    assert q3.knn_predict(np.array([[2.0], [4.0]]), np.array([20, 10]), query, 2)[0] == 10


def test_q3_selects_smallest_k_on_equal_perfect_accuracy() -> None:
    _, X_train, y_train = q3.load_csv(EXAM / "data" / "q3_train.csv", has_label=True)
    _, X_validation, y_validation = q3.load_csv(EXAM / "data" / "q3_validation.csv", has_label=True)
    assert y_train is not None and y_validation is not None
    selected, accuracy, records = q3.select_k(X_train, y_train, X_validation, y_validation)
    assert selected == 1 and accuracy == pytest.approx(1.0)
    assert records == [(1, 1.0), (3, 1.0), (5, 1.0)]


def test_q3_model_round_trip_shapes_and_values(tmp_path: Path) -> None:
    _, X_train, y_train = q3.load_csv(EXAM / "data" / "q3_train.csv", has_label=True)
    _, X_validation, y_validation = q3.load_csv(EXAM / "data" / "q3_validation.csv", has_label=True)
    assert y_train is not None and y_validation is not None
    model = q3.fit_pipeline(np.concatenate([X_train, X_validation]), np.concatenate([y_train, y_validation]))
    path = tmp_path / "model" / "pca_knn.npz"
    q3.save_model(path, model, 1)
    restored = q3.load_model(path)
    assert set(restored) == q3.MODEL_KEYS
    assert restored["components"].shape == (2, 3)
    assert restored["train_projection"].shape == (12, 2)
    for key in q3.MODEL_KEYS - {"selected_k"}:
        np.testing.assert_array_equal(restored[key], model[key])
    assert int(restored["selected_k"]) == 1


def run_q3(tmp_path: Path, test_path: Path, suffix: str):
    model = tmp_path / f"{suffix}.npz"
    predictions = tmp_path / f"{suffix}.csv"
    metrics = tmp_path / f"{suffix}.txt"
    q3.run(
        EXAM / "data" / "q3_train.csv",
        EXAM / "data" / "q3_validation.csv",
        test_path,
        model,
        predictions,
        metrics,
    )
    return q3.load_model(model), predictions, metrics


def test_q3_full_outputs_are_byte_exact(tmp_path: Path) -> None:
    _, predictions, metrics = run_q3(tmp_path, EXAM / "data" / "q3_test.csv", "base")
    assert predictions.read_bytes() == (EXAM / "expected" / "q3_predictions.csv").read_bytes()
    assert metrics.read_bytes() == (EXAM / "expected" / "q3_metrics.txt").read_bytes()


def test_q3_changed_test_does_not_change_model(tmp_path: Path) -> None:
    changed = tmp_path / "changed.csv"
    changed.write_text("sample_id,feature_1,feature_2,feature_3\nZ,999,999,999\n", encoding="utf-8")
    base, _, _ = run_q3(tmp_path, EXAM / "data" / "q3_test.csv", "base")
    altered, _, _ = run_q3(tmp_path, changed, "altered")
    for key in base:
        np.testing.assert_array_equal(base[key], altered[key])


def test_q3_rejects_explicit_nan_and_all_missing_fit_column(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("sample_id,feature_1,feature_2,feature_3\nX,nan,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="显式"):
        q3.load_csv(bad, has_label=False)
    with pytest.raises(ValueError, match="整列"):
        q3.fit_preprocessor(np.array([[1.0, np.nan, 2.0], [2.0, np.nan, 3.0]]))


def test_q3_does_not_modify_input_arrays() -> None:
    _, X, y = q3.load_csv(EXAM / "data" / "q3_train.csv", has_label=True)
    assert y is not None
    X_before, y_before = X.copy(), y.copy()
    model = q3.fit_pipeline(X, y)
    q3.knn_predict(model["train_projection"], model["train_labels"], model["train_projection"][:1], 1)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_student_files_remain_unimplemented() -> None:
    names = ("q1_group_summary", "q2_dijkstra", "q3_pca_knn")
    for name in names:
        path = EXAM / "student" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"student_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(NotImplementedError):
            if name == "q1_group_summary" or name == "q2_dijkstra":
                module.run(Path("in"), Path("out"))
            else:
                module.run(*([Path("x")] * 6))


def test_grader_awards_reference_full_score() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAM / "grade_submission.py"), "--submission", str(EXAM / "reference")],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "Q1=25/25" in completed.stdout
    assert "Q2=45/45" in completed.stdout
    assert "Q3=80/80" in completed.stdout
    assert "TOTAL=150/150" in completed.stdout
