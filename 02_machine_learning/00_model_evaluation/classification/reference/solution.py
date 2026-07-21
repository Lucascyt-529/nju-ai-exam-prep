"""参考适配层：复用父专题已经验证的分类指标实现。"""
import importlib.util
from pathlib import Path

def _load_core():
    path = Path(__file__).resolve().parents[2] / "reference" / "solution.py"
    spec = importlib.util.spec_from_file_location("model_evaluation_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

_CORE = _load_core()
def binary_confusion_matrix(y_true, y_pred, positive_label=1):
    return _CORE.binary_confusion_counts(y_true, y_pred, positive_label)
def accuracy_score(y_true, y_pred, positive_label=1):
    return _CORE.binary_classification_metrics(y_true, y_pred, positive_label)["accuracy"]
def precision_score(y_true, y_pred, positive_label=1):
    return _CORE.binary_classification_metrics(y_true, y_pred, positive_label)["precision"]
def recall_score(y_true, y_pred, positive_label=1):
    return _CORE.binary_classification_metrics(y_true, y_pred, positive_label)["recall"]
def f1_score(y_true, y_pred, positive_label=1):
    return _CORE.binary_classification_metrics(y_true, y_pred, positive_label)["f1"]
def binary_roc_curve(y_true, scores, positive_label=1):
    return _CORE.roc_curve_points(y_true, scores, positive_label)
def auc_trapezoid(fpr, tpr):
    return _CORE.auc_trapezoid(fpr, tpr)
