"""引导演示：按列中心化、主成分投影和重构。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("pca_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    pca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pca)

    X = np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])
    model = pca.fit_pca(X, 1)
    Z = pca.transform_pca(X, model["mean"], model["components"])
    reconstructed = pca.inverse_transform_pca(Z, model["mean"], model["components"])

    print("X shape:", X.shape)
    print("mean shape/value:", model["mean"].shape, model["mean"].tolist())
    print("covariance shape:\n", model["covariance"].shape, model["covariance"])
    print("components shape/value:", model["components"].shape, model["components"].tolist())
    print("Z shape:", Z.shape)
    print("explained variance ratio:", np.round(model["explained_variance_ratio"], 6).tolist())
    print("reconstruction mse:", round(pca.reconstruction_mse(X, reconstructed), 6))


if __name__ == "__main__":
    main()
