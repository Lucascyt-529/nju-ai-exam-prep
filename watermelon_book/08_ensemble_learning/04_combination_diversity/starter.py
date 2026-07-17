"""学生练习：集成结合策略与多样性度量。"""

import numpy as np


def weighted_average(predictions: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    raise NotImplementedError("请完成 weighted_average")


def hard_vote(predictions: np.ndarray) -> np.ndarray:
    raise NotImplementedError("请完成 hard_vote")


def pairwise_contingency(y: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> dict[str, int]:
    raise NotImplementedError("请完成 pairwise_contingency")


def regression_error_ambiguity(y: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 regression_error_ambiguity")


def bootstrap_index_matrix(n_samples: int, n_learners: int, *,
                           random_state: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 bootstrap_index_matrix")


def random_feature_subspaces(n_features: int, n_learners: int, subspace_size: int, *,
                             random_state: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 random_feature_subspaces")


def flipped_label_copies(y: np.ndarray, n_learners: int, *,
                         flip_fraction: float = 0.1,
                         random_state: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 flipped_label_copies")


def random_parameter_seeds(n_learners: int, *, random_state: int = 0) -> np.ndarray:
    raise NotImplementedError("请完成 random_parameter_seeds")


def diversity_perturbation_plan(n_samples: int, n_features: int, y: np.ndarray, *,
                                n_learners: int, subspace_size: int,
                                flip_fraction: float = 0.1,
                                random_state: int = 0) -> dict[str, np.ndarray]:
    raise NotImplementedError("请完成 diversity_perturbation_plan")
