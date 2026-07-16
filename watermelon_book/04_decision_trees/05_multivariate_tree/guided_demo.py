"""运行前比较单轴投影与斜线投影。"""

import numpy as np


def main() -> None:
    X = np.array([[-2, 1], [-1, 0], [0, -1], [-1, 2], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    axis = np.array([1.0, 0.0])
    oblique = np.array([1.0, 1.0]) / np.sqrt(2.0)
    print("X shape:", X.shape)
    print("labels:", y)
    print("axis projection:", X @ axis)
    print("oblique projection:", X @ oblique)
    print("oblique class-0 values:", (X @ oblique)[y == 0])
    print("oblique class-1 values:", (X @ oblique)[y == 1])


if __name__ == "__main__":
    main()
