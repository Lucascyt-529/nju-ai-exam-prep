"""运行前预测采样计数、类别权重和代价阈值。"""

import numpy as np


def main() -> None:
    y = np.array([0] * 90 + [1] * 10)
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    all_negative_accuracy = np.mean(y == 0)
    all_negative_recall = 0.0
    false_positive_cost = 1.0
    false_negative_cost = 4.0
    threshold = false_positive_cost / (false_positive_cost + false_negative_cost)

    print("class labels:", classes)
    print("class counts:", counts)
    print("balanced class weights:", weights)
    print("always-negative accuracy:", all_negative_accuracy)
    print("always-negative positive recall:", all_negative_recall)
    print("threshold when FP cost=1 and FN cost=4:", threshold)


if __name__ == "__main__":
    main()
