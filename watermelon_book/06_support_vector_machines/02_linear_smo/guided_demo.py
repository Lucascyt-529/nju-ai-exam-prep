"""训练一维线性SVM，检查alpha约束、支持向量和目标值。"""

import numpy as np

from reference.solution import (
    dual_objective,
    fit_linear_svm_smo,
    kkt_residuals,
    linear_kernel_matrix,
    linear_weights,
    predict_labels,
    primal_objective,
    support_vector_indices,
)


def main() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1, -1, 1, 1])
    model = fit_linear_svm_smo(X, y, C=10.0)
    gram = linear_kernel_matrix(X, X)
    dual = dual_objective(model["alphas"], y, gram)

    print("alphas:", np.round(model["alphas"], 6))
    print("sum alpha*y:", round(float(model["alphas"] @ y), 12))
    print("weights / bias:", np.round(linear_weights(model), 6), round(model["bias"], 6))
    print("support indices:", support_vector_indices(model))
    print("prediction:", predict_labels(model, X))
    print("max KKT residual:", round(float(np.max(kkt_residuals(model))), 12))
    print("primal / dual:", round(primal_objective(model), 6), round(dual, 6))


if __name__ == "__main__":
    main()
