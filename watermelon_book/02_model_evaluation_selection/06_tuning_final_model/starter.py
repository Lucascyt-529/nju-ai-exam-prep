"""学生练习：有限参数网格、验证集选择、最终重训与一次测试。"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np


Parameters = dict[str, object]


def cartesian_parameter_grid(
    choices: Mapping[str, Sequence[object]],
) -> list[Parameters]:
    raise NotImplementedError("请完成 cartesian_parameter_grid")


def tune_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    candidates: Sequence[Mapping[str, object]],
    fit_model: Callable[[np.ndarray, np.ndarray, Parameters], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    higher_is_better: bool,
) -> tuple[Parameters, list[dict[str, object]]]:
    raise NotImplementedError("请完成 tune_on_validation")


def refit_selected_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    selected_parameters: Mapping[str, object],
    fit_model: Callable[[np.ndarray, np.ndarray, Parameters], object],
) -> object:
    raise NotImplementedError("请完成 refit_selected_model")


def tune_refit_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    candidates: Sequence[Mapping[str, object]],
    fit_model: Callable[[np.ndarray, np.ndarray, Parameters], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    higher_is_better: bool,
) -> dict[str, object]:
    raise NotImplementedError("请完成 tune_refit_and_test")
