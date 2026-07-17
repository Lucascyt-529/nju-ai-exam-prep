"""引导演示：为什么模型文件必须同时保存预处理参数。"""

import importlib.util
from pathlib import Path
import tempfile

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("model_file_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    mean = np.array([10., 100.]); scale = np.array([2., 20.])
    weights = np.array([1.5, -0.5]); X = np.array([[12., 80.], [8., 140.]])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.data"
        module.save_model_bundle(path, mean, scale, weights, 0.25)
        restored = module.load_model_bundle(path)
        print("实际文件名:", path.name)
        print("模型键:", sorted(restored))
        print("mean shape:", restored["mean"].shape)
        print("bias shape in Python: scalar")
        print("predictions:", module.predict_with_bundle(X, restored).tolist())


if __name__ == "__main__":
    main()
