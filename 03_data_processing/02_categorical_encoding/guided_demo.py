"""引导演示：训练词表不应被测试集新类别改写。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("category_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    train = np.array([["红", "小"], ["蓝", "大"], ["红", "大"]])
    test = np.array([["绿", "小"], ["红", "中"]])
    vocabularies = module.fit_category_vocabularies(train)
    encoded = module.transform_one_hot(test, vocabularies, include_unknown=True)
    print("训练词表:", vocabularies)
    print("测试形状:", test.shape)
    print("one-hot形状:", encoded.shape)
    print("每个类别块行和均为1:", encoded[:, :3].sum(axis=1).tolist(), encoded[:, 3:].sum(axis=1).tolist())
    print("测试新类别没有进入训练词表:", "绿" not in vocabularies[0] and "中" not in vocabularies[1])


if __name__ == "__main__":
    main()
