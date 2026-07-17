"""参考实现：二值贝叶斯网校验、联合概率与枚举推断。"""

from itertools import product
import numpy as np


def _validate_parents(parents: dict[str, tuple[str, ...]]) -> None:
    if not isinstance(parents, dict) or not parents or any(not isinstance(name, str) or not name for name in parents):
        raise ValueError("parents必须是非空变量名字典")
    variables = set(parents)
    for variable, parent_tuple in parents.items():
        if not isinstance(parent_tuple, tuple) or len(set(parent_tuple)) != len(parent_tuple):
            raise ValueError("每个父节点列表必须是无重复元组")
        if variable in parent_tuple or any(parent not in variables for parent in parent_tuple):
            raise ValueError("父节点必须属于网络且不能包含自身")


def topological_order(parents: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    _validate_parents(parents)
    indegree = {node: len(values) for node, values in parents.items()}
    children = {node: [] for node in parents}
    for child, values in parents.items():
        for parent in values:
            children[parent].append(child)
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        node = ready.pop(0); order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child); ready.sort()
    if len(order) != len(parents):
        raise ValueError("网络包含有向环")
    return tuple(order)


def build_binary_network(parents: dict[str, tuple[str, ...]], probability_one: dict[str, np.ndarray]) -> dict[str, object]:
    order = topological_order(parents)
    if not isinstance(probability_one, dict) or set(probability_one) != set(parents):
        raise ValueError("probability_one必须恰好包含所有变量")
    copied = {}
    for variable, parent_tuple in parents.items():
        table = probability_one[variable]
        expected_shape = (2,) * len(parent_tuple)
        if not isinstance(table, np.ndarray) or table.shape != expected_shape or not np.issubdtype(table.dtype, np.number) or not np.all(np.isfinite(table)) or np.any((table < 0) | (table > 1)):
            raise ValueError(f"{variable}的CPT必须形状为{expected_shape}且概率位于[0,1]")
        copied[variable] = table.astype(float, copy=True)
    return {"parents": {node: tuple(values) for node, values in parents.items()}, "probability_one": copied, "order": order}


def _validate_model(model: dict[str, object]) -> None:
    if not isinstance(model, dict) or set(model) != {"parents", "probability_one", "order"}:
        raise ValueError("model键集合无效")
    if topological_order(model["parents"]) != model["order"]:
        raise ValueError("model拓扑顺序无效")


def _validate_values(values: dict[str, int], variables: set[str], *, complete: bool) -> None:
    if not isinstance(values, dict) or (set(values) != variables if complete else not set(values).issubset(variables)):
        raise ValueError("赋值变量集合无效")
    if any(value not in (0, 1) or isinstance(value, (bool, np.bool_)) for value in values.values()):
        raise ValueError("二值变量只能取整数0或1")


def joint_probability(model: dict[str, object], assignment: dict[str, int]) -> float:
    _validate_model(model)
    variables = set(model["parents"]); _validate_values(assignment, variables, complete=True)
    probability = 1.0
    for variable in model["order"]:
        indices = tuple(assignment[parent] for parent in model["parents"][variable])
        p_one = float(model["probability_one"][variable][indices])
        probability *= p_one if assignment[variable] == 1 else 1.0 - p_one
    return float(probability)


def all_assignments(model: dict[str, object]):
    _validate_model(model)
    for values in product((0, 1), repeat=len(model["order"])):
        yield dict(zip(model["order"], values))


def evidence_probability(model: dict[str, object], evidence: dict[str, int]) -> float:
    _validate_model(model); _validate_values(evidence, set(model["parents"]), complete=False)
    total = 0.0
    for assignment in all_assignments(model):
        if all(assignment[name] == value for name, value in evidence.items()):
            total += joint_probability(model, assignment)
    return float(total)


def query_posterior(model: dict[str, object], query: str, evidence: dict[str, int]) -> np.ndarray:
    _validate_model(model)
    if query not in model["parents"] or query in evidence:
        raise ValueError("query必须属于网络且不能已出现在evidence中")
    _validate_values(evidence, set(model["parents"]), complete=False)
    masses = np.array([evidence_probability(model, {**evidence, query: value}) for value in (0, 1)])
    if np.sum(masses) <= 0:
        raise ValueError("evidence概率为0，条件后验未定义")
    return masses / np.sum(masses)
