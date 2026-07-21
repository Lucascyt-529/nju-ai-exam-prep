"""展示线性核的输入和 Gram 矩阵。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected = np.eye(2)
    print("期望 Gram matrix:")
    print(expected)
    try:
        actual = starter.linear_kernel_matrix(X, X)
    except NotImplementedError:
        print("实际 Gram matrix: linear_kernel_matrix 尚未完成")
        return
    print("实际 Gram matrix:")
    print(actual)
    print("一致:", np.allclose(actual, expected))


if __name__ == "__main__":
    main()
