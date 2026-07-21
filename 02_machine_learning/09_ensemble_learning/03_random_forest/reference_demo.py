"""引导演示：随机候选特征与树桩选择。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("rf_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
    x = np.arange(12.0); X = np.column_stack((x, x[::-1], np.sin(x), np.cos(x)))
    y = np.where((x < 3) | ((x >= 6) & (x < 9)), -1, 1)
    model = rf.fit_random_subspace_forest(X, y, n_estimators=12, max_features=2, random_state=5)
    chosen = [learner.get("feature", None) for learner in model["learners"]]
    predictions = rf.base_predictions(model, X)
    print("feature subsets shape:", model["feature_subsets"].shape)
    print("first five subsets:", model["feature_subsets"][:5].tolist())
    print("chosen features:", chosen)
    print("mean prediction correlation:", round(rf.mean_pairwise_prediction_correlation(predictions), 4))
    print("ensemble prediction:", rf.predict(model, X).tolist())

if __name__ == "__main__": main()
