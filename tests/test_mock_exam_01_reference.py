import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAM = ROOT / "mock_exams" / "01_foundation_diagnostic"


def load_module(name: str):
    path = EXAM / "reference" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mock1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


q1 = load_module("q1_csv_stats")
q2 = load_module("q2_grid_bfs")
q3 = load_module("q3_logistic")


def test_q1_statistics_match_hand_calculation() -> None:
    X = q1.load_csv(EXAM / "data" / "q1_features.csv")
    means, stds, missing = q1.summarize(X)
    np.testing.assert_allclose(means, [3.0, 27.5, 8.0])
    np.testing.assert_allclose(stds, [np.sqrt(2), np.sqrt(175), 2.0])
    np.testing.assert_array_equal(missing, [1, 1, 1])


def test_q1_output_is_byte_exact(tmp_path: Path) -> None:
    output = tmp_path / "new" / "summary.csv"
    q1.run(EXAM / "data" / "q1_features.csv", output)
    assert output.read_bytes() == (EXAM / "expected" / "q1_summary.csv").read_bytes()


def test_q1_rejects_duplicate_and_all_missing_column(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "sample_id,feature_1,feature_2,feature_3\nA,1,2,3\nA,4,5,6\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="重复"):
        q1.load_csv(duplicate)
    with pytest.raises(ValueError, match="整列缺失"):
        q1.summarize(np.array([[1., np.nan, 2.], [2., np.nan, 3.]]))


def test_q2_sample_path_is_deterministic_and_byte_exact(tmp_path: Path) -> None:
    output = tmp_path / "path.txt"
    q2.run(EXAM / "data" / "q2_grid.txt", output)
    assert output.read_bytes() == (EXAM / "expected" / "q2_path.txt").read_bytes()


def test_q2_direction_order_breaks_shortest_path_tie() -> None:
    route = q2.shortest_path(["..", ".."], (1, 1), (0, 0))
    assert route == [(1, 1), (0, 1), (0, 0)]


def test_q2_start_goal_and_unreachable_boundaries() -> None:
    assert q2.shortest_path(["."], (0, 0), (0, 0)) == [(0, 0)]
    assert q2.shortest_path([".#", "#."], (0, 0), (1, 1)) is None


def test_q2_rejects_coordinate_on_obstacle(tmp_path: Path) -> None:
    source = tmp_path / "bad.txt"
    source.write_text("1 2\n.#\n0 0\n0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="障碍"):
        q2.load_problem(source)


def test_q3_preprocessing_uses_training_statistics_and_shapes() -> None:
    _, X, y = q3.load_csv(EXAM / "data" / "q3_train.csv", has_label=True)
    assert X.shape == (8, 3) and y is not None and y.shape == (8,)
    mean, std = q3.fit_preprocessor(X); scaled = q3.transform(X, mean, std)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-12)


def test_q3_sigmoid_is_stable_at_extreme_logits() -> None:
    probability = q3.sigmoid(np.array([-1000., 0., 1000.]))
    assert np.all(np.isfinite(probability))
    np.testing.assert_allclose(probability, [0., 0.5, 1.0], atol=1e-15)


def test_q3_gradient_matches_centered_finite_difference() -> None:
    X = np.array([[1., 2.], [-1., 0.5], [0.3, -2.]])
    y = np.array([1., 0., 1.]); weights = np.array([0.2, -0.4]); bias = 0.1; l2 = 0.03
    _, gradient, gradient_bias = q3.loss_and_gradient(X, y, weights, bias, l2=l2)
    epsilon = 1e-6; numerical = np.empty_like(weights)
    for index in range(len(weights)):
        plus = weights.copy(); minus = weights.copy(); plus[index] += epsilon; minus[index] -= epsilon
        numerical[index] = (
            q3.loss_and_gradient(X, y, plus, bias, l2=l2)[0]
            - q3.loss_and_gradient(X, y, minus, bias, l2=l2)[0]
        ) / (2 * epsilon)
    numerical_bias = (
        q3.loss_and_gradient(X, y, weights, bias + epsilon, l2=l2)[0]
        - q3.loss_and_gradient(X, y, weights, bias - epsilon, l2=l2)[0]
    ) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-7)
    assert gradient_bias == pytest.approx(numerical_bias, abs=1e-7)


def test_q3_training_loss_falls_and_validation_is_correct() -> None:
    _, X_train, y_train = q3.load_csv(EXAM / "data" / "q3_train.csv", has_label=True)
    _, X_validation, y_validation = q3.load_csv(EXAM / "data" / "q3_validation.csv", has_label=True)
    assert y_train is not None and y_validation is not None
    mean, std = q3.fit_preprocessor(X_train)
    model = q3.train(q3.transform(X_train, mean, std), y_train)
    assert model["loss_history"][-1] < model["loss_history"][0]
    probabilities = q3.predict_probability(q3.transform(X_validation, mean, std), model["weights"], model["bias"])
    assert q3.classification_metrics(y_validation, probabilities) == {"accuracy": 1.0, "f1": 1.0}


def test_q3_full_cli_outputs_are_byte_exact(tmp_path: Path) -> None:
    model = tmp_path / "model" / "q3.npz"; predictions = tmp_path / "out" / "pred.csv"; metrics = tmp_path / "out" / "metrics.txt"
    completed = subprocess.run(
        [
            sys.executable, str(EXAM / "reference" / "q3_logistic.py"),
            "--train", str(EXAM / "data" / "q3_train.csv"),
            "--validation", str(EXAM / "data" / "q3_validation.csv"),
            "--test", str(EXAM / "data" / "q3_test.csv"),
            "--model", str(model), "--predictions", str(predictions), "--metrics", str(metrics),
        ], check=False, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert predictions.read_bytes() == (EXAM / "expected" / "q3_predictions.csv").read_bytes()
    assert metrics.read_bytes() == (EXAM / "expected" / "q3_metrics.txt").read_bytes()
    assert model.is_file()


def test_q3_model_round_trip_and_test_isolation(tmp_path: Path) -> None:
    changed = tmp_path / "changed.csv"
    changed.write_text(
        "sample_id,feature_1,feature_2,feature_3\nZ,999,999,999\n", encoding="utf-8"
    )
    saved = []
    for index, test_path in enumerate((EXAM / "data" / "q3_test.csv", changed)):
        model_path = tmp_path / f"model{index}.npz"
        q3.run(
            EXAM / "data" / "q3_train.csv", EXAM / "data" / "q3_validation.csv", test_path,
            model_path, tmp_path / f"pred{index}.csv", tmp_path / f"metric{index}.txt",
        )
        saved.append(q3.load_model(model_path))
    for first, second in zip(saved[0][:-1], saved[1][:-1], strict=True):
        np.testing.assert_array_equal(first, second)
    assert saved[0][-1] == saved[1][-1]


def test_q3_f1_zero_denominator_boundary() -> None:
    assert q3.classification_metrics(np.zeros(3), np.zeros(3)) == {"accuracy": 1.0, "f1": 0.0}


def test_student_files_remain_unimplemented() -> None:
    for name in ("q1_csv_stats", "q2_grid_bfs", "q3_logistic"):
        spec = importlib.util.spec_from_file_location(f"student_{name}", EXAM / "student" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with pytest.raises(NotImplementedError):
            if name == "q1_csv_stats":
                module.run(Path("in"), Path("out"))
            elif name == "q2_grid_bfs":
                module.run(Path("in"), Path("out"))
            else:
                module.run(Path("train"), Path("validation"), Path("test"), Path("model"), Path("pred"), Path("metrics"))


def test_grader_awards_reference_submission_full_score() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAM / "grade_submission.py"), "--submission", str(EXAM / "reference")],
        check=False, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "Q1=30/30" in completed.stdout
    assert "Q2=45/45" in completed.stdout
    assert "Q3=75/75" in completed.stdout
    assert "TOTAL=150/150" in completed.stdout
