"""学生练习：从CSV读取到参数恢复和结果保存的完整预处理程序。"""

from pathlib import Path

import numpy as np


def load_feature_csv(
    path: Path, *, has_label: bool
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """读取固定两列特征；缺失特征保留为 NaN。"""
    raise NotImplementedError("请完成 load_feature_csv")


def fit_preprocessor(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回训练集拟合的 fill_values、means 和安全 scales。"""
    raise NotImplementedError("请完成 fit_preprocessor")


def transform_features(
    X: np.ndarray,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """使用已有参数填补并标准化，不重新拟合。"""
    raise NotImplementedError("请完成 transform_features")


def save_parameters(
    path: Path,
    fill_values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> None:
    """把三个参数数组保存到一个 .npz 文件。"""
    raise NotImplementedError("请完成 save_parameters")


def load_parameters(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 .npz 恢复并验证三个参数数组。"""
    raise NotImplementedError("请完成 load_parameters")


def save_transformed_csv(
    path: Path, sample_ids: list[str], X_transformed: np.ndarray
) -> None:
    """按固定表头和6位小数保存测试特征。"""
    raise NotImplementedError("请完成 save_transformed_csv")


def run_pipeline(train_path: Path, test_path: Path, output_path: Path, params_path: Path) -> None:
    """组织训练拟合、参数恢复和测试变换。"""
    raise NotImplementedError("请完成 run_pipeline")


def main() -> int:
    """解析命令行参数并处理可预期的数据错误。"""
    raise NotImplementedError("请完成 main")


if __name__ == "__main__":
    raise SystemExit(main())
