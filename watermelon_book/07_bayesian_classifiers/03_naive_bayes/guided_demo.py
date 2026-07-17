"""引导演示：离散未知值与高斯方差下限。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("nb_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    nb = importlib.util.module_from_spec(spec); spec.loader.exec_module(nb)
    Xd = np.array([["sunny", "hot"], ["sunny", "mild"], ["rainy", "mild"], ["rainy", "cool"]])
    y = np.array([0, 0, 1, 1])
    categorical = nb.fit_categorical_nb(Xd, y)
    query = np.array([["cloudy", "mild"]])
    print("categorical scores shape:", nb.joint_log_scores(categorical, query).shape)
    print("unseen-value posterior:", np.round(nb.predict_proba(categorical, query), 4))
    Xg = np.array([[0.0, 5.0], [0.2, 5.0], [2.0, 5.0], [2.2, 5.0]])
    gaussian = nb.fit_gaussian_nb(Xg, y, variance_floor=1e-4)
    print("gaussian means shape:", gaussian["means"].shape)
    print("gaussian variances:", gaussian["variances"])
    print("gaussian prediction:", nb.predict(gaussian, np.array([[2.1, 5.0]])).tolist())

if __name__ == "__main__": main()
