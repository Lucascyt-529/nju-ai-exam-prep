"""引导演示：MLE方差与无偏样本方差的分母差异。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("mle_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    mle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mle)
    samples = np.array([1.0, 2.0, 3.0])
    mean, variance = mle.gaussian_mle(samples)
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    vector_mean, covariance = mle.multivariate_gaussian_mle(X)
    print("mean:", mean)
    print("MLE variance ddof=0:", round(variance, 6))
    print("unbiased variance ddof=1:", round(float(np.var(samples, ddof=1)), 6))
    print("vector mean shape:", vector_mean.shape)
    print("covariance shape:", covariance.shape)
    print("covariance rank:", int(np.linalg.matrix_rank(covariance)))


if __name__ == "__main__":
    main()
