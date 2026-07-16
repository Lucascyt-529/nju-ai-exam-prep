"""运行前预测留出、分层K折和自助采样的索引性质。"""

import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)
    indices = rng.permutation(10)
    test = indices[:2]
    train = indices[2:]
    print("holdout train:", train)
    print("holdout test:", test)
    print("overlap:", np.intersect1d(train, test))

    folds = np.array_split(np.arange(10), 5)
    print("five validation folds:")
    for fold in folds:
        print(fold)

    bootstrap = rng.integers(0, 10, size=10)
    out_of_bag = np.setdiff1d(np.arange(10), np.unique(bootstrap))
    print("bootstrap:", bootstrap)
    print("out of bag:", out_of_bag)


if __name__ == "__main__":
    main()
