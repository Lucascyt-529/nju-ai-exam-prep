import importlib.util
from pathlib import Path
import numpy as np

def test_metrics_and_auc():
    path = Path(__file__).resolve().parents[1] / "reference" / "solution.py"
    spec = importlib.util.spec_from_file_location("classification_reference", path)
    solution = importlib.util.module_from_spec(spec); spec.loader.exec_module(solution)
    y = np.array([0, 0, 1, 1]); pred = np.array([0, 1, 1, 0])
    assert solution.binary_confusion_matrix(y, pred) == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert solution.f1_score(y, pred) == 0.5
    fpr, tpr, _ = solution.binary_roc_curve(y, np.array([0.1, 0.4, 0.35, 0.8]))
    assert np.isclose(solution.auc_trapezoid(fpr, tpr), 0.75)
