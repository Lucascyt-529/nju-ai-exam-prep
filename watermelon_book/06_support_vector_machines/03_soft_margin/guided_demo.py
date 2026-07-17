"""引导演示：比较C变化前后的软间隔分量和样本状态。"""

import importlib.util
from pathlib import Path

import numpy as np


TOPIC = Path(__file__).resolve().parent
SMO_SOLUTION = TOPIC.parent / "02_linear_smo" / "reference" / "solution.py"
SOFT_SOLUTION = TOPIC / "reference" / "solution.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    smo = _load("soft_margin_demo_smo", SMO_SOLUTION)
    soft = _load("soft_margin_demo_analysis", SOFT_SOLUTION)
    X = np.array([[-3.0], [-2.0], [-1.0], [0.2], [1.0], [2.0]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    print("scores、margins、slacks始终都是(n,)；不要把y改成(n,1)。")
    for C in (0.05, 0.2, 10.0):
        model = smo.fit_linear_svm_smo(
            X, y, C=C, tolerance=1e-6, max_passes=30, max_iterations=5000
        )
        scores = smo.decision_function(model, X)
        weights = smo.linear_weights(model)
        report = soft.analyze_soft_margin_solution(
            weights, y, scores, model["alphas"], C, tolerance=1e-5
        )
        print(f"\nC={C:g}")
        print("weights:", np.round(weights, 4))
        print("margins:", np.round(report["margins"], 4))
        print("slacks:", np.round(report["slacks"], 4))
        print("regions:", report["regions"].tolist())
        print("alpha status:", report["alpha_status"].tolist())
        print(
            "regularization/slack penalty/objective:",
            round(report["regularization"], 4),
            round(report["slack_penalty"], 4),
            round(report["objective"], 4),
        )


if __name__ == "__main__":
    main()
