"""展示阈值、混淆计数和 ROC/AUC 的关系。"""
import numpy as np
import starter

def main() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.10, 0.40, 0.35, 0.80, 0.70, 0.20])
    print("y_true:", y_true)
    print("scores:", scores)
    try:
        for threshold in (0.3, 0.5):
            y_pred = (scores >= threshold).astype(int)
            counts = starter.binary_confusion_matrix(y_true, y_pred)
            print(f"threshold={threshold}: y_pred={y_pred}, counts={counts}")
            print(" precision=", starter.precision_score(y_true, y_pred), "recall=", starter.recall_score(y_true, y_pred))
        fpr, tpr, thresholds = starter.binary_roc_curve(y_true, scores)
        print("ROC thresholds:", thresholds)
        print("FPR:", fpr)
        print("TPR:", tpr)
        print("AUC:", starter.auc_trapezoid(fpr, tpr))
    except NotImplementedError as error:
        print("停止展示：", error, sep="")
    except Exception as error:
        print(f"运行到当前步骤时出错：{type(error).__name__}: {error}")

if __name__ == "__main__":
    main()
