"""展示 sigmoid 的输入、期望概率和实际概率。"""

import numpy as np

import starter


def main() -> None:
    logits = np.array([-2.0, 0.0, 2.0])
    expected = np.array([0.11920292, 0.5, 0.88079708])
    print("logits:", logits)
    print("期望 probabilities:", expected)
    try:
        actual = starter.stable_sigmoid(logits)
    except NotImplementedError:
        print("实际 probabilities: stable_sigmoid 尚未完成")
        return
    print("实际 probabilities:", actual)
    print("一致:", np.allclose(actual, expected))


if __name__ == "__main__":
    main()
