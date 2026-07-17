"""引导演示：距离经双中心化和特征分解恢复坐标。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("mds_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    mds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mds)

    X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    distances = mds.pairwise_euclidean(X)
    gram = mds.double_center(distances)
    full = mds.classical_mds(distances, 2)
    reduced = mds.classical_mds(distances, 1)
    row_means = gram.mean(axis=1)
    row_means[np.abs(row_means) < 5e-13] = 0.0

    print("distance shape:", distances.shape)
    print("gram row means:", np.round(row_means, 12).tolist())
    print("eigenvalues:", np.round(full["eigenvalues"], 6).tolist())
    print("coordinates shape:", full["coordinates"].shape)
    print("full stress:", round(mds.normalized_stress(distances, full["coordinates"]), 12))
    print("one-dimensional stress:", round(mds.normalized_stress(distances, reduced["coordinates"]), 6))


if __name__ == "__main__":
    main()
