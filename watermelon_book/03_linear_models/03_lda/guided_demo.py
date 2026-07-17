"""手算二分类方向，再观察多分类散度与K-1维上限。"""

import numpy as np

from reference.solution import (
    fit_multiclass_lda,
    multiclass_scatter_matrices,
    predict_multiclass_lda,
)


def main() -> None:
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    mean0 = X[y == 0].mean(axis=0)
    mean1 = X[y == 1].mean(axis=0)
    centered0 = X[y == 0] - mean0
    centered1 = X[y == 1] - mean1
    scatter = centered0.T @ centered0 + centered1.T @ centered1
    weights = np.linalg.pinv(scatter) @ (mean1 - mean0)
    threshold = 0.5 * ((mean0 @ weights) + (mean1 @ weights))
    print("mean0:", mean0)
    print("mean1:", mean1)
    print("within-class scatter:")
    print(scatter)
    print("weights:", weights)
    print("threshold:", threshold)
    print("projections:", X @ weights)

    X_multi = np.array(
        [
            [-2.2, 0.2], [-2.0, 0.0], [-1.8, -0.2],
            [2.2, 0.2], [2.0, 0.0], [1.8, -0.2],
            [0.2, 3.2], [0.0, 3.0], [-0.2, 2.8],
        ]
    )
    y_multi = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30])
    classes, global_mean, means, within, between, total = (
        multiclass_scatter_matrices(X_multi, y_multi)
    )
    classes, projection, projected_centroids, eigenvalues = fit_multiclass_lda(
        X_multi, y_multi, regularization=1e-6
    )
    prediction = predict_multiclass_lda(
        X_multi, classes, projection, projected_centroids
    )
    print("multiclass classes:", classes)
    print("global/class means shapes:", global_mean.shape, means.shape)
    print("St equals Sw + Sb:", np.allclose(total, within + between))
    print("projection shape (at most K-1):", projection.shape)
    print("generalized eigenvalues:", np.round(eigenvalues, 6))
    print("training prediction:", prediction)


if __name__ == "__main__":
    main()
