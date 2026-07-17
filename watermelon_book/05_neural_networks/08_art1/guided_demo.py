"""比较ART1在低警戒值和高警戒值下的类别粒度。"""

import numpy as np

from reference.solution import train_art1


def main() -> None:
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ]
    )
    low_prototypes, low_assignments, low_history = train_art1(X, vigilance=0.4)
    high_prototypes, high_assignments, high_history = train_art1(X, vigilance=0.75)

    print("X:", X.shape)
    print("low vigilance:", low_prototypes.shape, low_assignments, low_history)
    print(low_prototypes)
    print("high vigilance:", high_prototypes.shape, high_assignments, high_history)
    print(high_prototypes)


if __name__ == "__main__":
    main()
