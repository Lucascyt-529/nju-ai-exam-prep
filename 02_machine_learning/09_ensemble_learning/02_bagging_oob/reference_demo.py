"""引导演示：Bootstrap重复、OOB覆盖与Bagging投票。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("bagging_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    bag = importlib.util.module_from_spec(spec); spec.loader.exec_module(bag)
    X = np.arange(8.0).reshape(-1, 1); y = np.array([-1, -1, 1, -1, 1, 1, -1, 1])
    model = bag.fit_bagging_stumps(X, y, n_estimators=20, random_state=7)
    scores, counts = bag.oob_decision_function(model, X); accuracy, covered = bag.oob_accuracy(model, X, y)
    print("bootstrap shape:", model["bootstrap_indices"].shape)
    print("first bootstrap:", model["bootstrap_indices"][0].tolist())
    print("first OOB:", bag.out_of_bag_indices(model["bootstrap_indices"][0], len(X)).tolist())
    print("OOB counts:", counts.tolist())
    print("covered:", int(np.sum(covered)), "/", len(X))
    print("OOB accuracy:", round(accuracy, 4))
    print("ensemble prediction:", bag.predict(model, X).tolist())

if __name__ == "__main__": main()
