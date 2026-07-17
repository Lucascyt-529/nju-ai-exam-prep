"""运行前预测RBF距离、隐藏响应和输出形状。"""

import numpy as np

from reference.solution import fit_rbf_output, predict_rbf, rbf_design_matrix


def main() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 0.0])
    centers = X.copy()
    design = rbf_design_matrix(X, centers, width=0.4)
    model = fit_rbf_output(X, y, centers, width=0.4)
    prediction = predict_rbf(model, X)

    print("X / y:", X.shape, y.shape)
    print("centers:", centers.shape)
    print("design:", design.shape)
    print("output weights:", model["output_weights"].shape)
    print("prediction:", prediction.shape, np.round(prediction, 6))


if __name__ == "__main__":
    main()
