"""引导演示：Cloudy-Sprinkler-Rain-WetGrass网络。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("bn_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    bn = importlib.util.module_from_spec(spec); spec.loader.exec_module(bn)
    parents = {"Cloudy": (), "Sprinkler": ("Cloudy",), "Rain": ("Cloudy",), "WetGrass": ("Sprinkler", "Rain")}
    cpt = {"Cloudy": np.array(0.5), "Sprinkler": np.array([0.5, 0.1]), "Rain": np.array([0.2, 0.8]), "WetGrass": np.array([[0.0, 0.9], [0.9, 0.99]])}
    model = bn.build_binary_network(parents, cpt)
    total = sum(bn.joint_probability(model, assignment) for assignment in bn.all_assignments(model))
    print("topological order:", model["order"])
    print("assignment count:", len(list(bn.all_assignments(model))))
    print("joint sum:", round(total, 10))
    print("P(WetGrass=1):", round(bn.evidence_probability(model, {"WetGrass": 1}), 6))
    print("P(Rain|WetGrass=1):", np.round(bn.query_posterior(model, "Rain", {"WetGrass": 1}), 6))

if __name__ == "__main__": main()
