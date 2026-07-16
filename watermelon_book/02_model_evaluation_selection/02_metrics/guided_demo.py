"""运行前手算回归误差、混淆计数和ROC曲线。"""

import numpy as np


def main() -> None:
    regression_true = np.array([1.0, 2.0, 3.0])
    regression_pred = np.array([1.0, 4.0, 2.0])
    errors = regression_pred - regression_true
    print("errors:", errors)
    print("MAE:", np.mean(np.abs(errors)))
    print("MSE:", np.mean(errors**2))

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("scores:", scores)
    print("TP:", np.count_nonzero((y_true == 1) & (y_pred == 1)))
    print("FP:", np.count_nonzero((y_true == 0) & (y_pred == 1)))
    print("TN:", np.count_nonzero((y_true == 0) & (y_pred == 0)))
    print("FN:", np.count_nonzero((y_true == 1) & (y_pred == 0)))


if __name__ == "__main__":
    main()
