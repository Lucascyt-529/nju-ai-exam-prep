"""分类评估日常核对。"""
import numpy as np
import starter

def compare(name, actual, expected):
    ok = actual == expected if isinstance(expected, dict) else bool(np.allclose(actual, expected))
    print(f"{name}: 期望={expected}, 实际={actual}, 通过={ok}")
    return ok

def main() -> int:
    y_true = np.array([0, 0, 1, 1]); y_pred = np.array([0, 1, 1, 0]); passed = 0
    try:
        passed += compare("混淆矩阵", starter.binary_confusion_matrix(y_true, y_pred), {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        passed += compare("accuracy", starter.accuracy_score(y_true, y_pred), 0.5)
        passed += compare("precision", starter.precision_score(y_true, y_pred), 0.5)
        passed += compare("recall", starter.recall_score(y_true, y_pred), 0.5)
        passed += compare("F1", starter.f1_score(y_true, y_pred), 0.5)
        passed += compare("零分母 precision", starter.precision_score(y_true, np.zeros(4, dtype=int)), 0.0)
        scores = np.array([0.1, 0.4, 0.35, 0.8]); fpr, tpr, thresholds = starter.binary_roc_curve(y_true, scores)
        passed += compare("ROC 起点", [fpr[0], tpr[0], thresholds[0]], [0.0, 0.0, np.inf])
        passed += compare("AUC", starter.auc_trapezoid(fpr, tpr), 0.75)
    except NotImplementedError as error:
        print("停止核对：", error, sep="")
    except Exception as error:
        print(f"运行错误：{type(error).__name__}: {error}")
    print(f"通过: {passed}/8")
    return 0 if passed == 8 else 1

if __name__ == "__main__":
    raise SystemExit(main())
