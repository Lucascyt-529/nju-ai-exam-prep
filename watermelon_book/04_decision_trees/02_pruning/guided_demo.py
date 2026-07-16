"""运行前判断一次划分在验证集上是否值得保留。"""

import numpy as np


def main() -> None:
    X_train = np.array([[0], [0], [1], [1]], dtype=float)
    y_train = np.array([0, 0, 1, 1])
    X_validation = np.array([[0], [0], [1], [1]], dtype=float)
    y_validation = np.array([0, 0, 0, 1])

    parent_prediction = 0
    split_predictions = np.where(X_validation[:, 0] == 0, 0, 1)
    parent_accuracy = np.mean(y_validation == parent_prediction)
    split_accuracy = np.mean(y_validation == split_predictions)

    print("training class counts:", np.unique(y_train, return_counts=True)[1])
    print("parent validation accuracy:", parent_accuracy)
    print("one-level split validation accuracy:", split_accuracy)
    print("keep split:", split_accuracy > parent_accuracy)


if __name__ == "__main__":
    main()
