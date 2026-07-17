"""参考实现：二值贝叶斯网校验、学习、评分、精确与Gibbs推断。"""

from itertools import product
from numbers import Real
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


def _validate_binary_data(data: np.ndarray, variable_names: tuple[str, ...]) -> None:
    if (not isinstance(data, np.ndarray) or data.ndim != 2 or 0 in data.shape
            or not np.all(np.isin(data, (0, 1)))):
        raise ValueError("data必须是非空二值二维数组")
    if (not isinstance(variable_names, tuple) or len(variable_names) != data.shape[1]
            or len(set(variable_names)) != len(variable_names)
            or any(not isinstance(name, str) or not name for name in variable_names)):
        raise ValueError("variable_names必须是与列数一致的无重复非空字符串元组")


def fit_binary_cpts(data: np.ndarray, variable_names: tuple[str, ...],
                    parents: dict[str, tuple[str, ...]], *, alpha: float = 0.0) -> dict[str, object]:
    """在给定DAG结构下按计数学习二值CPT；alpha为可选Laplace强度。"""
    _validate_binary_data(data, variable_names)
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("alpha必须是有限非负数")
    topological_order(parents)
    if set(parents) != set(variable_names):
        raise ValueError("parents与variable_names必须包含相同变量")
    column = {name: index for index, name in enumerate(variable_names)}
    probability_one = {}
    for variable, parent_names in parents.items():
        table = np.empty((2,) * len(parent_names), dtype=float)
        configurations = product((0, 1), repeat=len(parent_names)) if parent_names else [()]
        for configuration in configurations:
            mask = np.ones(len(data), dtype=bool)
            for parent, value in zip(parent_names, configuration):
                mask &= data[:, column[parent]] == value
            count = int(np.sum(mask))
            ones = int(np.sum(data[mask, column[variable]]))
            denominator = count + 2.0 * float(alpha)
            table[configuration] = (ones + float(alpha)) / denominator if denominator > 0 else 0.5
        probability_one[variable] = table
    return build_binary_network(parents, probability_one)


def parameter_count(model: dict[str, object]) -> int:
    """二值结点每种父配置只需一个自由参数。"""
    _validate_model(model)
    return int(sum(2 ** len(model["parents"][node]) for node in model["order"]))


def data_log_likelihood(model: dict[str, object], data: np.ndarray,
                        variable_names: tuple[str, ...]) -> float:
    _validate_model(model); _validate_binary_data(data, variable_names)
    if set(variable_names) != set(model["parents"]):
        raise ValueError("数据列与模型变量不一致")
    column = {name: index for index, name in enumerate(variable_names)}
    total = 0.0
    for row in data:
        assignment = {name: int(row[column[name]]) for name in model["order"]}
        probability = joint_probability(model, assignment)
        if probability <= 0:
            return float("-inf")
        total += np.log(probability)
    return float(total)


def bic_score(model: dict[str, object], data: np.ndarray,
              variable_names: tuple[str, ...]) -> float:
    """返回需最小化的负对数似然加BIC复杂度惩罚。"""
    log_likelihood = data_log_likelihood(model, data, variable_names)
    return float(-log_likelihood + 0.5 * parameter_count(model) * np.log(len(data)))


def greedy_ordered_structure(data: np.ndarray, variable_names: tuple[str, ...], *,
                             order: tuple[str, ...] | None = None,
                             max_parents: int = 2) -> dict[str, object]:
    """从空图出发，只按给定拓扑顺序向前加边并贪心降低BIC。"""
    _validate_binary_data(data, variable_names)
    if order is None:
        order = variable_names
    if (not isinstance(order, tuple) or len(order) != len(variable_names)
            or set(order) != set(variable_names)):
        raise ValueError("order必须是全部变量的一个元组排列")
    if (isinstance(max_parents, (bool, np.bool_)) or not isinstance(max_parents, (int, np.integer))
            or max_parents < 0):
        raise ValueError("max_parents必须是非负整数")
    parents = {name: () for name in variable_names}
    model = fit_binary_cpts(data, variable_names, parents)
    current_score = bic_score(model, data, variable_names)
    history = [current_score]
    position = {name: index for index, name in enumerate(order)}
    while True:
        best = None
        for parent in order:
            for child in order:
                if (position[parent] >= position[child] or parent in parents[child]
                        or len(parents[child]) >= max_parents):
                    continue
                candidate_parents = {name: tuple(values) for name, values in parents.items()}
                candidate_parents[child] = tuple(sorted((*candidate_parents[child], parent), key=position.get))
                candidate_model = fit_binary_cpts(data, variable_names, candidate_parents)
                candidate_score = bic_score(candidate_model, data, variable_names)
                key = (candidate_score, position[parent], position[child])
                if best is None or key < best[0]:
                    best = (key, candidate_parents, candidate_model)
        if best is None or best[0][0] >= current_score - 1e-12:
            break
        current_score = float(best[0][0]); parents = best[1]; model = best[2]
        history.append(current_score)
    return {"model": model, "parents": parents, "score_history": np.asarray(history)}


def _node_probability(model: dict[str, object], variable: str,
                      assignment: dict[str, int]) -> float:
    indices = tuple(assignment[parent] for parent in model["parents"][variable])
    probability_one = float(model["probability_one"][variable][indices])
    return probability_one if assignment[variable] == 1 else 1.0 - probability_one


def markov_blanket_probability_one(model: dict[str, object], variable: str,
                                   assignment: dict[str, int]) -> float:
    """固定其他变量时，根据本结点与直接子结点因子计算P(variable=1|-variable)。"""
    _validate_model(model); _validate_values(assignment, set(model["parents"]), complete=True)
    if variable not in model["parents"]:
        raise ValueError("variable不属于网络")
    children = [node for node in model["order"] if variable in model["parents"][node]]
    masses = []
    state = dict(assignment)
    for value in (0, 1):
        state[variable] = value
        mass = _node_probability(model, variable, state)
        for child in children:
            mass *= _node_probability(model, child, state)
        masses.append(mass)
    total = masses[0] + masses[1]
    if total <= 0:
        raise ValueError("当前Markov毯条件概率未定义")
    return float(masses[1] / total)


def gibbs_query(model: dict[str, object], query: str, evidence: dict[str, int], *,
                n_samples: int = 5000, burn_in: int = 500,
                random_state: int = 0) -> dict[str, object]:
    """固定证据，对其余二值结点逐个Gibbs采样并估计查询后验。"""
    _validate_model(model); _validate_values(evidence, set(model["parents"]), complete=False)
    if query not in model["parents"] or query in evidence:
        raise ValueError("query必须属于网络且不能是证据变量")
    for name, value in (("n_samples", n_samples), ("burn_in", burn_in)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or (value <= 0 if name == "n_samples" else value < 0)):
            raise ValueError("n_samples必须为正整数且burn_in必须为非负整数")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state必须是整数")
    if evidence_probability(model, evidence) <= 0:
        raise ValueError("evidence概率为0，条件后验未定义")
    rng = np.random.default_rng(int(random_state))
    state = {node: evidence.get(node, int(rng.integers(0, 2))) for node in model["order"]}
    sampled_variables = [node for node in model["order"] if node not in evidence]
    samples = np.empty(int(n_samples), dtype=int)
    saved = 0
    for sweep in range(int(burn_in) + int(n_samples)):
        for variable in sampled_variables:
            probability_one = markov_blanket_probability_one(model, variable, state)
            state[variable] = int(rng.random() < probability_one)
        if sweep >= burn_in:
            samples[saved] = state[query]; saved += 1
    posterior_one = float(np.mean(samples))
    return {"posterior": np.array([1.0 - posterior_one, posterior_one]),
            "query_samples": samples, "final_state": dict(state)}
