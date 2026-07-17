"""用均值回归器观察重复留出与重复K折的结果形状。"""

import numpy as np

from reference.solution import (
    evaluation_run_count,
    repeated_holdout_scores,
    repeated_kfold_scores,
    summarize_repeated_scores,
)


def fit_mean_regressor(X_train: np.ndarray, y_train: np.ndarray) -> float:
    return float(y_train.mean())


def predict_mean(model: object, X_test: np.ndarray) -> np.ndarray:
    return np.full(X_test.shape[0], float(model))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def main() -> None:
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + np.array(
        [0.0, 1.0, -1.0, 0.5, -0.5, 1.0, 0.0, -1.0, 0.5, 0.0, 1.0, -0.5]
    )

    holdout = repeated_holdout_scores(
        X,
        y,
        fit_mean_regressor,
        predict_mean,
        mse,
        repetitions=5,
        test_size=0.25,
        base_seed=42,
    )
    cross_validation = repeated_kfold_scores(
        X,
        y,
        fit_mean_regressor,
        predict_mean,
        mse,
        repetitions=3,
        n_splits=4,
        base_seed=42,
    )

    print("repeated holdout scores:", holdout["scores"])
    print("holdout summary:", summarize_repeated_scores(holdout["scores"]))
    print("repeated 4-fold score shape:", cross_validation["scores"].shape)
    print("scores by repetition:\n", cross_validation["scores"])
    print("repeat means:", summarize_repeated_scores(cross_validation["scores"])["repeat_means"])
    print("3 x 4 fit count:", evaluation_run_count(3, n_splits=4))
    print("10 x 10 fit count:", evaluation_run_count(10, n_splits=10))


if __name__ == "__main__":
    main()
