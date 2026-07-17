"""引导演示：选择次数很多，但测试集只在最终模型上评估一次。"""

import importlib.util
from pathlib import Path

import numpy as np


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"


def polynomial_fit(
    X: np.ndarray, y: np.ndarray, parameters: dict[str, object]
) -> dict[str, object]:
    degree = int(parameters["degree"])
    design = np.column_stack([X[:, 0] ** power for power in range(degree + 1)])
    weights = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"degree": degree, "weights": weights, "fit_sample_count": len(X)}


def polynomial_predict(model: object, X: np.ndarray) -> np.ndarray:
    if not isinstance(model, dict):
        raise TypeError("model必须是字典")
    design = np.column_stack(
        [X[:, 0] ** power for power in range(int(model["degree"]) + 1)]
    )
    return design @ model["weights"]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def main() -> None:
    spec = importlib.util.spec_from_file_location("tuning_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    tuning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuning)

    X_train = np.array([[-2.0], [-1.0], [0.0], [1.0]])
    y_train = X_train[:, 0] ** 2
    X_validation = np.array([[2.0], [3.0]])
    y_validation = X_validation[:, 0] ** 2
    X_test = np.array([[4.0], [5.0]])
    y_test = X_test[:, 0] ** 2
    candidates = tuning.cartesian_parameter_grid({"degree": [0, 1, 2]})

    result = tuning.tune_refit_and_test(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        candidates,
        polynomial_fit,
        polynomial_predict,
        mse,
        higher_is_better=False,
    )
    print("candidate count:", len(candidates))
    print("selected:", result["selected_parameters"])
    print("final fit sample count:", result["final_model"]["fit_sample_count"])
    print("test sample count excluded from fitting:", len(X_test))
    print("test score:", round(result["test_score"], 10))


if __name__ == "__main__":
    main()
