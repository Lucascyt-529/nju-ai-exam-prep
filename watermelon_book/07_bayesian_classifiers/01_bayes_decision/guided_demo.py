"""引导演示：最大后验与非对称代价下的最小风险决策。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("bayes_decision_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    bayes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bayes)
    priors = np.array([0.8, 0.2])
    likelihoods = np.array([[0.2, 0.3], [0.1, 0.9]])
    posteriors = bayes.posterior_from_likelihoods(priors, likelihoods)
    asymmetric = np.array([[0.0, 5.0], [1.0, 0.0]])
    print("posterior shape:", posteriors.shape)
    print("posteriors:", np.round(posteriors, 4))
    print("maximum posterior:", bayes.maximum_posterior_decisions(posteriors).tolist())
    print("conditional risks:", np.round(bayes.conditional_risks(posteriors, asymmetric), 4))
    print("minimum risk:", bayes.minimum_risk_decisions(posteriors, asymmetric).tolist())
    print("positive threshold:", round(bayes.binary_positive_threshold(1.0, 5.0), 4))


if __name__ == "__main__":
    main()
