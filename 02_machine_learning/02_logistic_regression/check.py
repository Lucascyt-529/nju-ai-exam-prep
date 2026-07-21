"""核对逻辑回归最前面的概率计算。"""

import numpy as np

import starter


def main() -> int:
    logits = np.array([-2.0, 0.0, 2.0])
    expected = np.array([0.11920292, 0.5, 0.88079708])
    try:
        actual = starter.stable_sigmoid(logits)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1

    correct = np.allclose(actual, expected)
    print("stable_sigmoid 期望:", expected)
    print("stable_sigmoid 实际:", actual)
    print("一致:", correct)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
