"""运行前先预测每个数组的 ndim、shape、size 和 dtype。"""

import numpy as np


def show(name: str, array: np.ndarray) -> None:
    print(
        name,
        "value=", array,
        "shape=", array.shape,
        "ndim=", array.ndim,
        "size=", array.size,
        "dtype=", array.dtype,
    )


def main() -> None:
    show("scalar", np.array(3.0))
    show("vector", np.array([1, 2, 3]))
    show("row", np.array([[1, 2, 3]], dtype=float))
    show("column", np.array([[1], [2], [3]], dtype=float))
    show("zeros", np.zeros((2, 3), dtype=float))
    show("sequence", np.arange(1, 8, 2, dtype=float))


if __name__ == "__main__":
    main()
