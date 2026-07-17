"""观察训练集IQR边界、异常掩码、裁剪和稳健缩放。"""

import numpy as np

from reference.solution import (
    clip_outliers,
    detect_outliers,
    fit_iqr_bounds,
    fit_robust_scaler,
    summarize_outliers,
    transform_robust_scaler,
)


def main() -> None:
    X_train = np.array(
        [
            [1.0, 10.0, 5.0],
            [2.0, 11.0, 5.0],
            [3.0, 12.0, 5.0],
            [4.0, 13.0, 5.0],
            [100.0, 14.0, 5.0],
        ]
    )
    X_test = np.array([[200.0, 12.0, 5.0], [3.0, -20.0, 8.0]])

    lower, upper = fit_iqr_bounds(X_train)
    train_mask = detect_outliers(X_train, lower, upper)
    test_mask = detect_outliers(X_test, lower, upper)
    feature_counts, row_flags, row_count = summarize_outliers(test_mask)
    clipped_test = clip_outliers(X_test, lower, upper)
    medians, scales = fit_robust_scaler(X_train)
    scaled_test = transform_robust_scaler(clipped_test, medians, scales)

    print("train/test shapes:", X_train.shape, X_test.shape)
    print("lower bounds:", lower)
    print("upper bounds:", upper)
    print("train outlier mask first column:", train_mask[:, 0])
    print("test feature outlier counts:", feature_counts)
    print("test rows with outliers:", row_flags, "total:", row_count)
    print("clipped test:\n", clipped_test)
    print("training medians:", medians)
    print("safe IQR scales:", scales)
    print("scaled test:\n", scaled_test)


if __name__ == "__main__":
    main()
