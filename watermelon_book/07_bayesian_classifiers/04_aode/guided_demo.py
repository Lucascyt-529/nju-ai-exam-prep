"""引导演示：合格超父、AODE分数与朴素回退。"""

import importlib.util
from pathlib import Path
import numpy as np

SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("aode_demo", SOLUTION)
    if spec is None or spec.loader is None: raise RuntimeError("无法加载参考实现")
    aode = importlib.util.module_from_spec(spec); spec.loader.exec_module(aode)
    X = np.array([["red", "round", "sweet"], ["red", "round", "sweet"], ["green", "long", "sour"], ["green", "long", "sour"], ["red", "long", "sour"], ["green", "round", "sweet"]])
    y = np.array([1, 1, 0, 0, 0, 1])
    model = aode.fit_aode(X, y, min_parent_count=2)
    query = np.array([["red", "round", "sweet"], ["blue", "square", "bitter"]])
    print("eligible parents first:", aode.eligible_parent_indices(model, query[0]).tolist())
    print("eligible parents unseen:", aode.eligible_parent_indices(model, query[1]).tolist())
    print("AODE scores:", np.round(aode.joint_log_scores(model, query), 4))
    print("naive fallback unseen:", np.round(aode.naive_joint_log_scores(model, query[1:]), 4))
    print("posterior row sums:", np.round(aode.predict_proba(model, query).sum(axis=1), 4))

if __name__ == "__main__": main()
