"""运行前先预测OvR分数形状和OvO类别对数量。"""

import numpy as np


def main() -> None:
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.5, 0.5],
            [2.0, 0.0],
            [1.5, 0.5],
            [0.0, 3.0],
            [0.5, 2.5],
        ]
    )
    y = np.array([10, 10, 20, 20, 30, 30])
    classes = np.unique(y)
    ovr_targets = np.column_stack([y == label for label in classes]).astype(int)
    class_pairs = np.array(
        [(left, right) for left in range(len(classes)) for right in range(left + 1, len(classes))]
    )

    print("X shape:", X.shape)
    print("classes:", classes)
    print("OvR target matrix shape:", ovr_targets.shape)
    print(ovr_targets)
    print("OvO classifier count:", len(class_pairs))
    print("OvO internal class-index pairs:")
    print(class_pairs)
    print("OvO original-label pairs:")
    print(classes[class_pairs])


if __name__ == "__main__":
    main()
