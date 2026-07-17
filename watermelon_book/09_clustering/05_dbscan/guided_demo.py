"""引导演示：从邻域计数到核心点、边界点和噪声点。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("dbscan_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    dbscan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dbscan)

    X = np.array([[0.0], [0.2], [0.4], [0.6], [1.0], [3.0], [3.2], [3.4], [3.6]])
    model = dbscan.fit_dbscan(X, eps=0.21, min_samples=3)

    print("X shape:", X.shape)
    print("neighborhood shape:", model["neighborhood_matrix"].shape)
    print("neighbor counts:", model["neighbor_counts"].tolist())
    print("core indices:", np.flatnonzero(model["core_mask"]).tolist())
    print("border indices:", np.flatnonzero(model["border_mask"]).tolist())
    print("noise indices:", np.flatnonzero(model["noise_mask"]).tolist())
    print("labels:", model["labels"].tolist())
    print("clusters:", model["clusters"])


if __name__ == "__main__":
    main()
