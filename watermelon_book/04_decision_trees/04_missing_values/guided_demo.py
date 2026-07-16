"""运行前手算缺失样本的分支权重。"""

import numpy as np


def main() -> None:
    feature = np.array([0.0, 0.0, 1.0, np.nan])
    weights = np.array([1.0, 1.0, 1.0, 2.0])
    known = ~np.isnan(feature)
    values = np.unique(feature[known])
    known_masses = np.array([weights[known & (feature == value)].sum() for value in values])
    proportions = known_masses / known_masses.sum()
    print("known branch masses:", known_masses)
    print("branch proportions:", proportions)
    print("missing sample original weight:", weights[-1])
    print("missing sample propagated weights:", weights[-1] * proportions)
    print("propagated total:", np.sum(weights[-1] * proportions))


if __name__ == "__main__":
    main()
