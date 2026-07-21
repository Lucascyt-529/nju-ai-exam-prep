"""核对熵、信息增益和完整离散树。"""

import numpy as np

import starter


def main() -> int:
    X = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])
    passed = 0
    try:
        entropy = starter.entropy(y)
        gain = starter.information_gain(X[:, 0], y)
        tree = starter.fit_discrete_tree(X, y)
        prediction = starter.predict_discrete_tree(tree, X)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1

    passed += int(np.isclose(entropy, 1.0))
    passed += int(np.isclose(gain, 1.0))
    passed += int(np.array_equal(prediction, y))
    print("entropy 期望/实际: 1.0 /", entropy)
    print("information_gain 期望/实际: 1.0 /", gain)
    print("prediction 期望/实际:", y, "/", prediction)
    print(f"通过: {passed}/3")
    return 0 if passed == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
