"""展示训练集拟合、测试集复用参数的完整过程。"""

import numpy as np


def main() -> None:
    X_train = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    X_test = np.array([[100.0, np.nan], [200.0, 999.0]])

    fill_values = np.nanmean(X_train, axis=0)
    train_filled = np.where(np.isnan(X_train), fill_values, X_train)
    test_filled = np.where(np.isnan(X_test), fill_values, X_test)

    means = train_filled.mean(axis=0)
    stds = train_filled.std(axis=0)
    safe_stds = np.where(stds == 0, 1.0, stds)

    print("fill values from train:", fill_values)
    print("filled train =")
    print(train_filled)
    print("filled test =")
    print(test_filled)
    print("train means:", means)
    print("train safe stds:", safe_stds)
    print("scaled train mean:", ((train_filled - means) / safe_stds).mean(axis=0))
    print("scaled test mean:", ((test_filled - means) / safe_stds).mean(axis=0))


if __name__ == "__main__":
    main()
