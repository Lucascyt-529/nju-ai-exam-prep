"""引导演示：边际相同但条件依赖不同的二值数据。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("tan_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    tan = importlib.util.module_from_spec(spec); spec.loader.exec_module(tan)
    X = np.tile(np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=int), (20, 1))
    y = np.tile(np.array([0, 0, 1, 1], dtype=int), 20)
    model = tan.fit_tan(X, y, root=0, alpha=1.0)
    print("conditional MI:", round(model["weights"][0, 1], 6))
    print("tree parents:", model["parents"].tolist())
    print("score shape:", tan.tan_log_scores(model, X[:4]).shape)
    print("predictions:", tan.predict_tan(model, X[:4]).tolist())
    print("training accuracy:", float(np.mean(tan.predict_tan(model, X) == y)))


if __name__ == "__main__": main()
