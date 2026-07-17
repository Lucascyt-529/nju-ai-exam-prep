"""引导演示：每个epoch覆盖样本一次，同时保持X/y同步。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("minibatch_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    X = np.column_stack((np.arange(7), np.arange(7) * 10)).astype(float)
    y = np.arange(7).astype(float) + 100
    schedule = module.batch_index_schedule(7, 3, 2, random_state=4)
    for epoch, batches in enumerate(schedule, start=1):
        print(f"epoch {epoch}:", [batch.tolist() for batch in batches])
        print("  覆盖:", sorted(np.concatenate(batches).tolist()))
    indices = schedule[0][0]
    X_batch, y_batch = module.take_minibatch(X, y, indices)
    print("首批下标:", indices.tolist())
    print("首批X第一列:", X_batch[:, 0].astype(int).tolist())
    print("首批y减100:", (y_batch - 100).astype(int).tolist())
    print("X/y同步:", np.array_equal(X_batch[:, 0], y_batch - 100))


if __name__ == "__main__":
    main()
