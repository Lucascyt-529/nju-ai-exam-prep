"""展示配对差异、McNemar不一致计数与多数据集秩。"""

import numpy as np


def main() -> None:
    scores_a = np.array([0.82, 0.78, 0.85, 0.80, 0.84])
    scores_b = np.array([0.79, 0.77, 0.80, 0.81, 0.79])
    differences = scores_a - scores_b
    t_statistic = differences.mean() / (
        differences.std(ddof=1) / np.sqrt(differences.size)
    )
    print("paired differences:", differences)
    print("paired t statistic:", t_statistic)

    y_true = np.array([1, 1, 1, 0, 0, 0])
    prediction_a = np.array([1, 1, 0, 0, 1, 0])
    prediction_b = np.array([1, 0, 1, 0, 0, 1])
    a_wrong_b_right = np.count_nonzero(
        (prediction_a != y_true) & (prediction_b == y_true)
    )
    a_right_b_wrong = np.count_nonzero(
        (prediction_a == y_true) & (prediction_b != y_true)
    )
    print("A错B对:", a_wrong_b_right)
    print("A对B错:", a_right_b_wrong)


if __name__ == "__main__":
    main()
