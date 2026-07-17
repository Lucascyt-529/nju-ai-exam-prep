"""引导演示：同一组带符号间隔在三种替代损失下的不同作用。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("surrogate_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    losses = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses)

    margins = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    print("margins:", margins.tolist())
    print("zero-one:", losses.zero_one_losses(margins).tolist())
    for kind in ("hinge", "exponential", "logistic"):
        values = losses.surrogate_losses(margins, kind=kind)
        active = losses.active_gradient_mask(margins, kind=kind)
        print(kind, "losses:", np.round(values, 6).tolist())
        print(kind, "active gradient:", active.tolist())
    print("hinge ignores margin>1 samples exactly; smooth losses do not")


if __name__ == "__main__":
    main()
