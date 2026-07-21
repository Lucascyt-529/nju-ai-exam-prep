"""按主线核对距离和分类结果。"""

import numpy as np

import starter


def main() -> int:
    X_train = np.array([[0.0], [1.0], [3.0], [4.0]])
    y_train = np.array(["A", "A", "B", "B"])
    X_query = np.array([[0.5], [3.5]])
    try:
        distances = starter.pairwise_euclidean(X_query, X_train)
        prediction = starter.predict_classification(X_query, X_train, y_train, 2)
    except NotImplementedError as error:
        print("停止核对:", error)
        return 1

    distance_ok = np.allclose(distances[0], [0.5, 0.5, 2.5, 3.5])
    prediction_ok = np.array_equal(prediction, ["A", "B"])
    print("第一行距离期望: [0.5 0.5 2.5 3.5]")
    print("第一行距离实际:", distances[0], "一致:", distance_ok)
    print("分类期望: [A B]")
    print("分类实际:", prediction, "一致:", prediction_ok)
    return 0 if distance_ok and prediction_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
