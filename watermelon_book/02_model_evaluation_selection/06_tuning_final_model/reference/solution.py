"""参考实现：有限参数网格、验证集选择、最终重训与一次测试。"""

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np


Parameters = dict[str, object]


def _validate_parameter_value(value: object, name: str) -> object:
    if not isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        raise ValueError(f"参数{name}的候选值必须是字符串、布尔或有限数值标量")
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"参数{name}的数值候选必须有限")
    return value.item() if isinstance(value, (np.integer, np.floating)) else value


def cartesian_parameter_grid(
    choices: Mapping[str, Sequence[object]],
) -> list[Parameters]:
    """按参数名排序，生成有限笛卡尔积候选配置。"""
    if not isinstance(choices, Mapping) or len(choices) == 0:
        raise ValueError("choices必须是非空参数候选映射")
    names = sorted(choices)
    value_lists: list[list[object]] = []
    for name in names:
        if not isinstance(name, str) or name == "":
            raise ValueError("参数名必须是非空字符串")
        values = choices[name]
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) == 0
        ):
            raise ValueError(f"参数{name}必须提供非空候选序列")
        checked = [_validate_parameter_value(value, name) for value in values]
        if len({repr(value) for value in checked}) != len(checked):
            raise ValueError(f"参数{name}的候选值不能重复")
        value_lists.append(checked)
    return [dict(zip(names, values, strict=True)) for values in product(*value_lists)]


def _supervised(X: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(X, np.ndarray)
        or not np.issubdtype(X.dtype, np.number)
        or X.ndim != 2
        or 0 in X.shape
        or not np.all(np.isfinite(X))
    ):
        raise ValueError(f"{name} X必须是非空有限数值二维数组")
    if (
        not isinstance(y, np.ndarray)
        or y.ndim != 1
        or y.shape[0] != X.shape[0]
        or y.size == 0
    ):
        raise ValueError(f"{name} y必须是一维数组且样本数与X一致")
    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        raise ValueError(f"{name}数值y必须只含有限值")
    return X, y


def _validate_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> None:
    train_X, _ = _supervised(X_train, y_train, "训练集")
    validation_X, _ = _supervised(X_validation, y_validation, "验证集")
    if train_X.shape[1] != validation_X.shape[1]:
        raise ValueError("训练集与验证集特征数必须一致")
    if (X_test is None) != (y_test is None):
        raise ValueError("X_test和y_test必须同时提供或同时省略")
    if X_test is not None and y_test is not None:
        test_X, _ = _supervised(X_test, y_test, "测试集")
        if test_X.shape[1] != train_X.shape[1]:
            raise ValueError("测试集与训练集特征数必须一致")


def _validated_candidates(candidates: Sequence[Mapping[str, object]]) -> list[Parameters]:
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or len(candidates) == 0
    ):
        raise ValueError("candidates必须是非空参数配置序列")
    checked: list[Parameters] = []
    expected_keys: set[str] | None = None
    signatures: set[tuple[tuple[str, str], ...]] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or len(candidate) == 0:
            raise ValueError(f"第{index + 1}个候选必须是非空参数映射")
        config: Parameters = {}
        for name, value in candidate.items():
            if not isinstance(name, str) or name == "":
                raise ValueError("参数名必须是非空字符串")
            config[name] = _validate_parameter_value(value, name)
        keys = set(config)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("所有候选配置必须包含相同参数名")
        signature = tuple((name, repr(config[name])) for name in sorted(config))
        if signature in signatures:
            raise ValueError("候选参数配置不能重复")
        signatures.add(signature)
        checked.append(config)
    return checked


def _prediction(
    predict: Callable[[object, np.ndarray], np.ndarray],
    model: object,
    X: np.ndarray,
) -> np.ndarray:
    values = np.asarray(predict(model, X))
    if values.ndim != 1 or values.shape[0] != X.shape[0]:
        raise ValueError("predict必须为每个样本返回一个一维预测")
    if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
        raise ValueError("数值预测必须只含有限值")
    return values


def _score(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    value = metric(y_true, y_pred)
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
    ):
        raise ValueError("metric必须返回有限数值标量")
    return float(value)


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
    """只用训练集拟合候选，只用验证集选择；同分时保留先出现者。"""
    _validate_splits(X_train, y_train, X_validation, y_validation)
    configs = _validated_candidates(candidates)
    records: list[dict[str, object]] = []
    best_index = 0
    best_score: float | None = None
    for index, config in enumerate(configs):
        model = fit_model(X_train, y_train, dict(config))
        predictions = _prediction(predict, model, X_validation)
        score = _score(metric, y_validation, predictions)
        records.append({"parameters": dict(config), "validation_score": score})
        if best_score is None or (
            score > best_score if higher_is_better else score < best_score
        ):
            best_index, best_score = index, score
    return dict(configs[best_index]), records


def refit_selected_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    selected_parameters: Mapping[str, object],
    fit_model: Callable[[np.ndarray, np.ndarray, Parameters], object],
) -> object:
    """参数选定后，用训练集与验证集合并的数据重新拟合最终模型。"""
    _validate_splits(X_train, y_train, X_validation, y_validation)
    config = _validated_candidates([selected_parameters])[0]
    X_development = np.concatenate([X_train, X_validation], axis=0)
    y_development = np.concatenate([y_train, y_validation], axis=0)
    return fit_model(X_development, y_development, dict(config))


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
    """完整执行调参、开发集重训和一次测试评估。"""
    _validate_splits(
        X_train, y_train, X_validation, y_validation, X_test, y_test
    )
    selected, records = tune_on_validation(
        X_train,
        y_train,
        X_validation,
        y_validation,
        candidates,
        fit_model,
        predict,
        metric,
        higher_is_better=higher_is_better,
    )
    final_model = refit_selected_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        selected,
        fit_model,
    )
    test_predictions = _prediction(predict, final_model, X_test)
    test_score = _score(metric, y_test, test_predictions)
    return {
        "selected_parameters": selected,
        "validation_records": records,
        "final_model": final_model,
        "test_predictions": test_predictions,
        "test_score": test_score,
    }
