"""运行前先预测OvR/OvO编码形状，再观察ECOC纠错。"""

import numpy as np

from reference.solution import (
    binary_error_correction_capacity,
    decode_ecoc,
    make_ovo_coding_matrix,
    make_ovr_coding_matrix,
    minimum_hamming_distance,
)


def main() -> None:
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.5, 0.5],
            [2.0, 0.0],
            [1.5, 0.5],
            [0.0, 3.0],
            [0.5, 2.5],
        ]
    )
    y = np.array([10, 10, 20, 20, 30, 30])
    classes = np.unique(y)
    ovr_targets = np.column_stack([y == label for label in classes]).astype(int)
    class_pairs = np.array(
        [(left, right) for left in range(len(classes)) for right in range(left + 1, len(classes))]
    )

    print("X shape:", X.shape)
    print("classes:", classes)
    print("OvR target matrix shape:", ovr_targets.shape)
    print(ovr_targets)
    print("OvO classifier count:", len(class_pairs))
    print("OvO internal class-index pairs:")
    print(class_pairs)
    print("OvO original-label pairs:")
    print(classes[class_pairs])
    print("OvR coding matrix:")
    print(make_ovr_coding_matrix(len(classes)))
    print("OvO ternary coding matrix:")
    print(make_ovo_coding_matrix(len(classes)))

    classes4 = np.array([10, 20, 30, 40])
    long_binary_code = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [-1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, -1, 1, -1, 1],
            [-1, -1, -1, 1, -1, 1, 1],
        ]
    )
    one_error_code = long_binary_code[2].copy()
    one_error_code[0] *= -1
    prediction = decode_ecoc(
        one_error_code.reshape(1, -1), classes4, long_binary_code
    )
    print("long binary code shape:", long_binary_code.shape)
    print("minimum Hamming distance:", minimum_hamming_distance(long_binary_code))
    print(
        "guaranteed binary correction bits:",
        binary_error_correction_capacity(long_binary_code),
    )
    print("class 30 code with one flipped bit:", one_error_code)
    print("decoded original label:", prediction[0])


if __name__ == "__main__":
    main()
