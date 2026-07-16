"""参考实现：二项、配对、McNemar、Friedman与Nemenyi比较。"""

import math

import numpy as np


def binomial_lower_tail(successes: int, trials: int, probability: float) -> float:
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes 和 trials 必须是整数")
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("必须满足 0 <= successes <= trials")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability 必须位于 [0, 1]")
    return float(
        sum(
            math.comb(trials, index)
            * probability**index
            * (1 - probability) ** (trials - index)
            for index in range(successes + 1)
        )
    )


def _paired_scores(
    scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError("两组分数必须是形状一致且至少含2项的一维数组")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("分数必须只包含有限数值")
    return a, b


def paired_t_statistic(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    a, b = _paired_scores(scores_a, scores_b)
    differences = a - b
    standard_deviation = differences.std(ddof=1)
    if standard_deviation == 0:
        return 0.0 if differences.mean() == 0 else math.copysign(math.inf, differences.mean())
    return float(differences.mean() / (standard_deviation / np.sqrt(differences.size)))


def corrected_resampled_t_statistic(
    differences: np.ndarray, n_train: int, n_validation: int
) -> float:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("differences 必须是至少含2项的有限一维数组")
    if not isinstance(n_train, int) or not isinstance(n_validation, int) or n_train <= 0 or n_validation <= 0:
        raise ValueError("训练和验证样本数必须是正整数")
    variance = values.var(ddof=1)
    correction = 1.0 / values.size + n_validation / n_train
    denominator = np.sqrt(correction * variance)
    if denominator == 0:
        return 0.0 if values.mean() == 0 else math.copysign(math.inf, values.mean())
    return float(values.mean() / denominator)


def mcnemar_disagreements(
    y_true: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray
) -> tuple[int, int]:
    truth = np.asarray(y_true)
    a = np.asarray(prediction_a)
    b = np.asarray(prediction_b)
    if truth.ndim != 1 or a.ndim != 1 or b.ndim != 1 or truth.size == 0:
        raise ValueError("真实标签和两个预测必须是非空一维数组")
    if truth.shape != a.shape or truth.shape != b.shape:
        raise ValueError("真实标签和两个预测形状必须一致")
    a_correct = a == truth
    b_correct = b == truth
    return (
        int(np.count_nonzero(~a_correct & b_correct)),
        int(np.count_nonzero(a_correct & ~b_correct)),
    )


def mcnemar_exact_p_value(b: int, c: int) -> float:
    if not isinstance(b, int) or not isinstance(c, int) or b < 0 or c < 0:
        raise ValueError("b 和 c 必须是非负整数")
    disagreements = b + c
    if disagreements == 0:
        return 1.0
    smaller = min(b, c)
    return min(1.0, 2.0 * binomial_lower_tail(smaller, disagreements, 0.5))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def ranks_per_dataset(
    scores: np.ndarray, *, higher_is_better: bool
) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("scores 必须是至少2个数据集×2个算法的二维数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores 必须只包含有限数值")
    ranked_values = -values if higher_is_better else values
    return np.vstack([_average_ranks(row) for row in ranked_values])


def friedman_statistic(ranks: np.ndarray) -> tuple[np.ndarray, float]:
    rank_array = np.asarray(ranks, dtype=float)
    if rank_array.ndim != 2 or rank_array.shape[0] < 2 or rank_array.shape[1] < 2:
        raise ValueError("ranks 必须是至少2×2的二维数组")
    if not np.all(np.isfinite(rank_array)):
        raise ValueError("ranks 必须只包含有限数值")
    n_datasets, n_algorithms = rank_array.shape
    expected_sum = n_algorithms * (n_algorithms + 1) / 2
    if not np.allclose(rank_array.sum(axis=1), expected_sum):
        raise ValueError("每个数据集的秩之和不合法")
    average_ranks = rank_array.mean(axis=0)
    statistic = (
        12 * n_datasets / (n_algorithms * (n_algorithms + 1))
        * (
            np.sum(average_ranks**2)
            - n_algorithms * (n_algorithms + 1) ** 2 / 4
        )
    )
    return average_ranks, float(statistic)


def nemenyi_critical_difference(
    n_algorithms: int, n_datasets: int, q_alpha: float
) -> float:
    if not isinstance(n_algorithms, int) or n_algorithms < 2:
        raise ValueError("n_algorithms 必须至少为2")
    if not isinstance(n_datasets, int) or n_datasets < 2:
        raise ValueError("n_datasets 必须至少为2")
    if not np.isfinite(q_alpha) or q_alpha <= 0:
        raise ValueError("q_alpha 必须是正有限数值")
    return float(
        q_alpha * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_datasets))
    )


def nemenyi_significant_pairs(
    names: list[str], average_ranks: np.ndarray, critical_difference: float
) -> list[tuple[str, str]]:
    ranks = np.asarray(average_ranks, dtype=float)
    if ranks.ndim != 1 or ranks.size != len(names) or ranks.size < 2:
        raise ValueError("names 和 average_ranks 必须长度一致且至少含2个算法")
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("算法名称必须非空且唯一")
    if not np.all(np.isfinite(ranks)) or not np.isfinite(critical_difference) or critical_difference < 0:
        raise ValueError("秩和临界差必须是合法有限数值")
    pairs: list[tuple[str, str]] = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            if abs(ranks[first] - ranks[second]) > critical_difference:
                pairs.append((names[first], names[second]))
    return pairs
