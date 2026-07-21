"""给出一组线性预测输入和期望输出，方便手工核对。"""

import numpy as np

import starter


def main() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    w = np.array([2.0])
    b = 1.0
    expected = np.array([3.0, 5.0, 7.0])

    print("输入 X:")
    print(X)
    print("输入 w:", w)
    print("输入 b:", b)
    print("期望 prediction:", expected)

    try:
        actual = starter.predict(X, w, b)
    except NotImplementedError:
        print("实际 prediction: predict 尚未完成")
        return

    print("实际 prediction:", actual)
    print("是否一致:", np.allclose(actual, expected))


if __name__ == "__main__":
    main()
