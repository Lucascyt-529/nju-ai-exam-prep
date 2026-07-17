"""引导演示：训练内预测为何不能作为Stacking次级训练特征。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


class SeenSampleModel:
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.seen = set(X[:, 0].tolist())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([float(value in self.seen) for value in X[:, 0]])


def main() -> None:
    spec = importlib.util.spec_from_file_location("stacking_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    stacking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stacking)

    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = X[:, 0] ** 2
    in_sample_model = SeenSampleModel().fit(X, y)
    in_sample = in_sample_model.predict(X)
    report = stacking.build_oof_meta_features(
        X, y, [SeenSampleModel], n_splits=4, shuffle=False
    )

    print("训练内预测:", in_sample.astype(int).tolist())
    print("折外预测:  ", report["meta_features"][:, 0].astype(int).tolist())
    print("每个样本只出现一次:", sorted(np.concatenate(report["validation_folds"]).tolist()) == list(range(len(X))))
    print("结论: 训练内全为1只是模型记住了样本，不能作为次级训练特征。")


if __name__ == "__main__":
    main()
