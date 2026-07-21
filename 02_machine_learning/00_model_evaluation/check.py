"""按顺序核对四个回归指标。"""

import numpy as np

import starter


def compare(name: str, actual: float, expected: float) -> bool:
    correct = bool(np.isclose(actual, expected))
    print(f"{name}: 期望={expected}, 实际={actual}, 一致={correct}")
    return correct


def main() -> int:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 2.0])
    passed = 0
    try:
        passed += compare("MAE", starter.mean_absolute_error(y_true, y_pred), 1.0)
        passed += compare("MSE", starter.mean_squared_error(y_true, y_pred), 5.0 / 3.0)
        passed += compare("RMSE", starter.root_mean_squared_error(y_true, y_pred), np.sqrt(5.0 / 3.0))
        passed += compare("R2", starter.r2_score(y_true, y_pred), -1.5)
    except NotImplementedError as error:
        print("停止核对:", error)

    print(f"通过: {passed}/4")
    return 0 if passed == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
