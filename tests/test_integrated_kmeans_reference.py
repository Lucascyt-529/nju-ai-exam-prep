import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOPIC = ROOT / "integrated_tasks" / "04_kmeans_csv"
SOLUTION = TOPIC / "reference" / "solution.py"
spec = importlib.util.spec_from_file_location("integrated_kmeans_solution", SOLUTION)
assert spec is not None and spec.loader is not None
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


def test_csv_loading_preserves_ids_shape_and_missing_values() -> None:
    train_ids, X_train = solution.load_feature_csv(TOPIC / "data" / "train.csv")
    test_ids, X_test = solution.load_feature_csv(TOPIC / "data" / "test.csv")
    assert train_ids[:2] == ["A1", "A2"] and X_train.shape == (9, 2)
    assert test_ids == ["T1", "T2", "T3"] and X_test.shape == (3, 2)
    assert np.isnan(X_test[1, 1]) and np.isnan(X_test[2, 0])


def test_preprocessor_uses_column_means_and_safe_zero_std() -> None:
    X = np.array([[1., 5.], [3., np.nan], [5., 5.]])
    mean, std = solution.fit_preprocessor(X)
    np.testing.assert_allclose(mean, [3., 5.])
    np.testing.assert_allclose(std, [np.sqrt(8 / 3), 1.])
    transformed = solution.transform_features(np.array([[np.nan, 5.]]), mean, std)
    np.testing.assert_allclose(transformed, [[0., 0.]])


def test_preprocessor_rejects_all_missing_training_column() -> None:
    with pytest.raises(ValueError, match="整列缺失"):
        solution.fit_preprocessor(np.array([[1., np.nan], [2., np.nan]]))


def test_one_kmeans_run_is_finite_and_canonical() -> None:
    _, X = solution.load_feature_csv(TOPIC / "data" / "train.csv")
    mean, std = solution.fit_preprocessor(X)
    scaled = solution.transform_features(X, mean, std)
    model = solution.fit_kmeans_once(scaled, 3, random_state=4)
    assert model["centers"].shape == (3, 2)
    assert model["labels"].shape == (9,)
    assert np.all(np.isfinite(model["centers"])) and model["inertia"] >= 0
    centers = model["centers"]
    assert all(tuple(centers[i]) <= tuple(centers[i + 1]) for i in range(2))


def test_silhouette_matches_small_hand_example() -> None:
    X = np.array([[0.], [2.], [10.], [12.]])
    labels = np.array([0, 0, 1, 1])
    # a=2；两个左簇样本的b分别为11和9，右簇对称。
    expected = np.mean([(11 - 2) / 11, (9 - 2) / 9, (9 - 2) / 9, (11 - 2) / 11])
    assert solution.silhouette_score(X, labels) == pytest.approx(expected)


def test_model_selection_finds_three_clusters_and_records_each_candidate() -> None:
    _, X = solution.load_feature_csv(TOPIC / "data" / "train.csv")
    mean, std = solution.fit_preprocessor(X)
    scaled = solution.transform_features(X, mean, std)
    model = solution.select_kmeans_model(scaled, [2, 3, 4], n_init=10, random_state=0)
    assert model["n_clusters"] == 3
    assert [item[0] for item in model["candidate_summary"]] == [2, 3, 4]
    assert model["silhouette"] == pytest.approx(0.966505075, rel=1e-6)
    assert sorted(np.bincount(model["labels"]).tolist()) == [3, 3, 3]


def test_more_initializations_cannot_worsen_same_k_best_inertia() -> None:
    _, X = solution.load_feature_csv(TOPIC / "data" / "train.csv")
    mean, std = solution.fit_preprocessor(X); scaled = solution.transform_features(X, mean, std)
    once = solution.select_kmeans_model(scaled, [3], n_init=1, random_state=8)
    repeated = solution.select_kmeans_model(scaled, [3], n_init=8, random_state=8)
    assert repeated["inertia"] <= once["inertia"] + 1e-12


def test_model_round_trip_uses_exact_path(tmp_path: Path) -> None:
    path = tmp_path / "cluster.model"
    mean = np.array([1., 2.]); std = np.array([3., 4.]); centers = np.array([[0., 1.], [2., 3.]])
    solution.save_model(path, mean, std, centers)
    restored = solution.load_model(path)
    assert path.is_file() and not (tmp_path / "cluster.model.npz").exists()
    for actual, expected in zip(restored, (mean, std, centers), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_full_command_line_outputs_are_byte_exact(tmp_path: Path) -> None:
    assignments = tmp_path / "out" / "assignments.csv"
    diagnostics = tmp_path / "out" / "diagnostics.txt"
    model = tmp_path / "model" / "kmeans.npz"
    completed = subprocess.run(
        [
            sys.executable, str(SOLUTION),
            "--train", str(TOPIC / "data" / "train.csv"),
            "--test", str(TOPIC / "data" / "test.csv"),
            "--model", str(model),
            "--assignments", str(assignments),
            "--diagnostics", str(diagnostics),
            "--candidate-k", "2", "3", "4",
            "--n-init", "10", "--random-state", "0",
        ],
        check=False, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert assignments.read_bytes() == (TOPIC / "expected" / "assignments.csv").read_bytes()
    assert diagnostics.read_bytes() == (TOPIC / "expected" / "diagnostics.txt").read_bytes()
    assert model.is_file()


def test_changing_test_data_cannot_change_fitted_model(tmp_path: Path) -> None:
    changed_test = tmp_path / "changed.csv"
    changed_test.write_text(
        "sample_id,feature_1,feature_2\nZ1,1000,1000\n", encoding="utf-8"
    )
    models = []
    for index, test_path in enumerate((TOPIC / "data" / "test.csv", changed_test)):
        model_path = tmp_path / f"model{index}.npz"
        solution.run_pipeline(
            TOPIC / "data" / "train.csv", test_path, model_path,
            tmp_path / f"assign{index}.csv", tmp_path / f"diag{index}.txt",
            random_state=5,
        )
        models.append(solution.load_model(model_path))
    for first, second in zip(models[0], models[1], strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("sample_id,feature_1\nA,1\n", "表头"),
        ("sample_id,feature_1,feature_2\nA,1,2\nA,2,3\n", "重复"),
        ("sample_id,feature_1,feature_2\nA,nan,2\n", "有限"),
        ("sample_id,feature_1,feature_2\nA,,\n", "所有特征"),
    ],
)
def test_invalid_csv_is_rejected(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.csv"; path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        solution.load_feature_csv(path)


def test_invalid_candidate_k_is_rejected() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError):
        solution.select_kmeans_model(X, [1, 2])
    with pytest.raises(ValueError):
        solution.select_kmeans_model(X, [2, 2])


def test_starter_is_kept_for_student_work() -> None:
    starter_spec = importlib.util.spec_from_file_location("integrated_kmeans_starter", TOPIC / "starter.py")
    assert starter_spec is not None and starter_spec.loader is not None
    starter = importlib.util.module_from_spec(starter_spec); starter_spec.loader.exec_module(starter)
    with pytest.raises(NotImplementedError):
        starter.load_feature_csv(TOPIC / "data" / "train.csv")
