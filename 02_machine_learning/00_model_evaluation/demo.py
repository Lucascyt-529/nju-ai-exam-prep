"""展示一组回归指标输入和期望输出。"""

import numpy as np

import starter


def main() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 2.0])
    expected = 5.0 / 3.0

    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("期望 MSE:", expected)
    try:
        actual = starter.mean_squared_error(y_true, y_pred)
    except NotImplementedError:
        print("实际 MSE: mean_squared_error 尚未完成")
        return
    print("实际 MSE:", actual)
    print("一致:", np.isclose(actual, expected))


if __name__ == "__main__":
    main()
