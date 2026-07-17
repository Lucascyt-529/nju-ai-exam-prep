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
    plan=d.diversity_perturbation_plan(6,5,y,n_learners=3,subspace_size=2,flip_fraction=1/3,random_state=9)
    print("sample index shape:",plan["sample_indices"].shape)
    print("feature subspaces:",plan["feature_subspaces"].tolist())
    print("flips per learner:",np.sum(plan["perturbed_labels"]!=y[None,:],axis=1).tolist())
    print("parameter seed shape:",plan["parameter_seeds"].shape)

if __name__ == "__main__": main()
