"""引导演示：距离矩阵的行列、最近邻及两类加权。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("knn_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    knn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(knn)

    X_train = np.array([[0.0], [1.0], [3.0], [4.0]])
    y_class = np.array(["A", "A", "B", "B"])
    y_reg = np.array([0.0, 1.0, 9.0, 16.0])
    X_query = np.array([[0.8], [3.2]])

    distances = knn.pairwise_euclidean(X_query, X_train)
    neighbor_distances, neighbor_indices = knn.kneighbors(X_query, X_train, 3)
    print("distance shape:", distances.shape)
    print("distance rows:\n", np.round(distances, 2))
    print("neighbor indices:", neighbor_indices.tolist())
    print("neighbor distances:", np.round(neighbor_distances, 2).tolist())
    print("uniform class:", knn.predict_classification(X_query, X_train, y_class, 3).tolist())
    print("distance class:", knn.predict_classification(X_query, X_train, y_class, 3, weights="distance").tolist())
    print("uniform regression:", np.round(knn.predict_regression(X_query, X_train, y_reg, 2), 3).tolist())


if __name__ == "__main__":
    main()
