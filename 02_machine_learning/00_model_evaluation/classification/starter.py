"""学生练习：二分类混淆矩阵、指标与 ROC/AUC。"""
import numpy as np

def binary_confusion_matrix(y_true, y_pred, positive_label=1) -> dict[str, int]:
    raise NotImplementedError("请完成 binary_confusion_matrix")
def accuracy_score(y_true, y_pred, positive_label=1) -> float:
    raise NotImplementedError("请完成 accuracy_score")
def precision_score(y_true, y_pred, positive_label=1) -> float:
    raise NotImplementedError("请完成 precision_score")
def recall_score(y_true, y_pred, positive_label=1) -> float:
    raise NotImplementedError("请完成 recall_score")
def f1_score(y_true, y_pred, positive_label=1) -> float:
    raise NotImplementedError("请完成 f1_score")
def binary_roc_curve(y_true, scores, positive_label=1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("请完成 binary_roc_curve")
def auc_trapezoid(fpr, tpr) -> float:
    raise NotImplementedError("请完成 auc_trapezoid")
