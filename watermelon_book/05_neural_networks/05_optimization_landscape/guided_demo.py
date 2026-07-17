"""比较双井函数的全部驻点和多个梯度下降初值。"""

from reference.solution import (
    classify_stationary_point,
    gradient_descent,
    objective,
    stationary_points,
)


def main() -> None:
    print("stationary points:")
    for point in stationary_points():
        print(round(point, 6), classify_stationary_point(point), round(objective(point), 6))

    print("descent endpoints:")
    for start in [-2.0, -0.2, 0.2, 2.0]:
        positions, values = gradient_descent(start)
        print(start, "->", round(positions[-1], 6), round(values[-1], 6))


if __name__ == "__main__":
    main()
