# 二分类评估

二分类模型先输出分数或概率，再由阈值得到0/1标签。指标回答的问题不同，类别不平衡时不能只看 accuracy。

## 混淆矩阵与公式

以标签1为正类：TP 是真实正且预测正，FP 是真实负但预测正，TN 是真实负且预测负，FN 是真实正但预测负。

```text
accuracy  = (TP + TN) / N
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

本专题约定分母为0时 precision、recall、F1 返回0。100个样本只有2个正类时，全部预测为负也有98% accuracy，但正类 recall 为0。

## 手算、阈值与 ROC

`y_true=[0,0,1,1]`、`y_pred=[0,1,1,0]` 时，TP/FP/TN/FN 都为1，四个指标都为0.5。降低概率阈值通常会预测更多正类，recall 可能提高而 precision 可能降低；阈值应在验证集选择。

ROC 遍历从 `+inf` 到各唯一分数的阈值，横轴是 `FPR=FP/(FP+TN)`，纵轴是 `TPR=recall`。`auc_trapezoid` 按相邻 ROC 点做梯形面积求和。

## 函数职责与实现顺序

```text
binary_confusion_matrix -> accuracy_score / precision_score / recall_score
-> f1_score -> binary_roc_curve -> auc_trapezoid
```

输入是一维真实标签与预测标签或连续分数，输出是计数字典、浮点数或三个一维数组。

```powershell
python 02_machine_learning/00_model_evaluation/classification/demo.py
python 02_machine_learning/00_model_evaluation/classification/check.py
```

常见错误：混淆 FP 与 FN、直接平均 precision 和 recall、用硬标签计算 AUC、同分数时逐样本生成 ROC 点，以及在测试集调阈值。

自学闭环：手算四样本，完成混淆矩阵和四指标；构造全负预测观察零分母；改变阈值并解释 precision/recall 的变化。
