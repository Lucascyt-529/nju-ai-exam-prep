"""运行前先写出相邻不同值的中点。"""

import numpy as np


def main() -> None:
    feature = np.array([1.0, 1.0, 3.0, 5.0])
    values = np.unique(feature)
    thresholds = (values[:-1] + values[1:]) / 2.0
    print("sorted unique values:", values)
    print("candidate thresholds:", thresholds)
    for threshold in thresholds:
        print(
            f"threshold {threshold}: left={np.sum(feature <= threshold)}, "
            f"right={np.sum(feature > threshold)}"
        )


if __name__ == "__main__":
    main()
