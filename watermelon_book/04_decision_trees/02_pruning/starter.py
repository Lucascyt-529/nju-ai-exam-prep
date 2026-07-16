"""学生练习：验证集预剪枝与后剪枝。"""

from typing import Any

import numpy as np


Tree = dict[str, Any]


def count_tree_nodes(tree: Tree) -> int:
    raise NotImplementedError("请完成 count_tree_nodes")


def fit_prepruned_tree(
    X: np.ndarray,
    y: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    criterion: str = "information_gain",
) -> Tree:
    raise NotImplementedError("请完成 fit_prepruned_tree")


def post_prune_tree(
    tree: Tree, X_validation: np.ndarray, y_validation: np.ndarray
) -> Tree:
    raise NotImplementedError("请完成 post_prune_tree")
