"""引导演示：XOR在线性核与RBF核下的差异。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("kernel_svm_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    svm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svm)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([-1, 1, 1, -1])
    for kernel, gamma in (("linear", None), ("rbf", 2.0)):
        model = svm.fit_kernel_svm_smo(X, y, C=10.0, kernel=kernel, gamma=gamma, tolerance=1e-6, max_passes=30)
        print(f"kernel={kernel}")
        print("train Gram shape:", svm.kernel_matrix(X, X, kernel=kernel, gamma=gamma).shape)
        print("support indices:", svm.support_vector_indices(model).tolist())
        print("scores:", np.round(svm.decision_function(model, X), 4))
        print("predictions:", svm.predict_labels(model, X).tolist())


if __name__ == "__main__":
    main()
