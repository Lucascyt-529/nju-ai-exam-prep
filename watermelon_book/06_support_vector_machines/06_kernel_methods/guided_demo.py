"""引导演示：表示定理形状与RBF-KLDA处理XOR。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("klda_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    klda = importlib.util.module_from_spec(spec); spec.loader.exec_module(klda)
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = klda.fit_kernel_lda(X, y, kernel="rbf", gamma=2.0, regularization=1e-4)
    query_kernel = klda.kernel_matrix(X, model["X_train"], kernel="rbf", gamma=2.0)
    print("query-train kernel shape:", query_kernel.shape)
    print("coefficient shape:", model["coefficients"].shape)
    print("scores:", np.round(klda.decision_function(model, X), 6))
    print("predictions:", klda.predict(model, X).tolist())
    print("fisher ratio finite or infinite:", bool(np.isfinite(klda.fisher_ratio(model)) or np.isinf(klda.fisher_ratio(model))))


if __name__ == "__main__": main()
