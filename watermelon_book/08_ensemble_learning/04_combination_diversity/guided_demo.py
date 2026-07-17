"""引导演示：四格多样性与回归误差-分歧恒等式。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("diversity_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
    y = np.array([1,1,-1,-1,1,-1]); a = np.array([1,1,-1,1,-1,-1]); b = np.array([1,-1,-1,-1,1,1])
    counts = d.pairwise_contingency(y,a,b)
    print("contingency:", counts); print("metrics:", d.diversity_metrics(counts))
    targets = np.array([1.,2.,3.]); predictions = np.array([[1.,2.5,2.5],[0.,2.,4.],[2.,1.5,3.]])
    report = d.regression_error_ambiguity(targets,predictions)
    print("ensemble:", report["ensemble_prediction"])
    print("identity holds:", np.allclose(report["ensemble_error"], report["mean_individual_error"]-report["ambiguity"]))

if __name__ == "__main__": main()
