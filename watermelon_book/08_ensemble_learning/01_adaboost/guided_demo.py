"""引导演示：首轮错分样本权重上升与多轮组合。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("adaboost_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    ada = importlib.util.module_from_spec(spec); spec.loader.exec_module(ada)
    X = np.arange(5.0).reshape(-1, 1); y = np.array([-1, 1, -1, 1, 1])
    model = ada.fit_adaboost(X, y, n_estimators=10)
    print("learner count:", len(model["learners"]))
    print("first stump:", model["learners"][0])
    print("weight history shape:", model["weight_history"].shape)
    print("initial weights:", np.round(model["weight_history"][0], 4))
    print("after first round:", np.round(model["weight_history"][1], 4))
    print("training scores:", np.round(ada.decision_function(model, X), 4))
    print("training prediction:", ada.predict(model, X).tolist())

if __name__ == "__main__": main()
