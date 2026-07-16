"""使用均值回归基线展示折内训练、验证和折外预测。"""

import numpy as np


def main() -> None:
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    validation_folds = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    all_indices = np.arange(len(y))
    out_of_fold = np.empty_like(y)

    for fold_number, validation_indices in enumerate(validation_folds, start=1):
        train_indices = np.setdiff1d(all_indices, validation_indices)
        training_mean = y[train_indices].mean()
        predictions = np.full(validation_indices.size, training_mean)
        out_of_fold[validation_indices] = predictions
        mse = np.mean((y[validation_indices] - predictions) ** 2)
        print(
            f"fold {fold_number}:",
            "train=", train_indices,
            "validation=", validation_indices,
            "mean=", training_mean,
            "mse=", mse,
        )

    print("out-of-fold predictions:", out_of_fold)
    print("overall MSE:", np.mean((y - out_of_fold) ** 2))


if __name__ == "__main__":
    main()
