"""比较超平面正比例缩放前后的函数间隔和几何间隔。"""

import numpy as np

from reference.solution import functional_margins, geometric_margins, minimum_margin_indices


def main() -> None:
    X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    y = np.array([-1, -1, 1, 1])
    weights = np.array([1.0, 0.0])
    bias = 0.0

    functional = functional_margins(X, y, weights, bias)
    geometric = geometric_margins(X, y, weights, bias)
    scaled_functional = functional_margins(X, y, 3.0 * weights, 3.0 * bias)
    scaled_geometric = geometric_margins(X, y, 3.0 * weights, 3.0 * bias)

    print("X / y / w:", X.shape, y.shape, weights.shape)
    print("functional:", functional)
    print("scaled functional:", scaled_functional)
    print("geometric:", geometric)
    print("scaled geometric:", scaled_geometric)
    print("minimum-margin indices:", minimum_margin_indices(X, y, weights, bias))


if __name__ == "__main__":
    main()
