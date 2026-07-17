"""学生练习：完成无OJ的K-means CSV聚类程序。"""

from pathlib import Path

import numpy as np


FEATURE_COLUMNS = ["feature_1", "feature_2"]


def load_feature_csv(path: Path) -> tuple[list[str], np.ndarray]:
    raise NotImplementedError("第1关：读取CSV并用NaN表示空白特征")


def fit_preprocessor(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("第2关：只用训练集拟合填补均值和安全标准差")


def transform_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raise NotImplementedError("第2关：复用训练集参数变换任意数据")


def fit_kmeans_once(X: np.ndarray, n_clusters: int, *, random_state: int = 0,
                    max_iterations: int = 100, tolerance: float = 1e-6) -> dict[str, object]:
    raise NotImplementedError("第3关：完成一次K-means++与迭代更新")


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    raise NotImplementedError("第4关：计算内部轮廓系数")


def select_kmeans_model(X: np.ndarray, candidate_ks: tuple[int, ...] | list[int], *,
                        n_init: int = 5, random_state: int = 0) -> dict[str, object]:
    raise NotImplementedError("第4关：多次初始化并用轮廓系数选k")


def save_model(path: Path, mean: np.ndarray, std: np.ndarray, centers: np.ndarray) -> None:
    raise NotImplementedError("第5关：保存预处理参数和聚类中心")


def load_model(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("第5关：安全恢复模型")


def run_pipeline(train_path: Path, test_path: Path, model_path: Path,
                 assignments_path: Path, diagnostics_path: Path, *,
                 candidate_ks: tuple[int, ...] = (2, 3, 4),
                 n_init: int = 10, random_state: int = 0) -> None:
    raise NotImplementedError("最终关：串起完整无OJ流程")
