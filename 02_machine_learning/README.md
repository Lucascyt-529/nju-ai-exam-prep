# 机器学习

## 当前学习入口

[01_linear_regression/README.md](01_linear_regression/README.md)

这里表示学习者现在继续填写的位置，不表示仓库只准备了线性回归。

## 学习顺序

```text
回归指标 -> 数据划分 -> 线性回归 -> 分类指标 -> 逻辑回归 -> kNN -> PCA -> K-means
-> 决策树 -> 朴素贝叶斯 -> 神经网络 -> 集成学习 -> LDA -> SVM
```

模型评估不是与模型分离的一门孤立练习。每个模型都要同时回答：如何划分数据、用什么指标比较、何时选择参数、测试集何时使用。

## 已准备的算法入口

| 顺序 | 专题 | 当前代表实现 |
| --- | --- | --- |
| 0 | [模型评估](00_model_evaluation/) | 回归指标、数据划分、分类指标与 ROC/AUC |
| 1 | [线性回归](01_linear_regression/) | 梯度下降与最小二乘 |
| 2 | [逻辑回归](02_logistic_regression/) | 二分类逻辑回归 |
| 3 | [kNN](03_knn/) | 欧氏距离分类 |
| 4 | [PCA](04_pca/) | 特征分解降维与重构 |
| 5 | [K-means](05_kmeans/) | 最近中心与迭代更新 |
| 6 | [决策树](06_decision_tree/) | 离散特征信息增益树 |
| 7 | [朴素贝叶斯](07_naive_bayes/) | 离散特征分类 |
| 8 | [神经网络入口](08_neural_network/) | 先从感知机开始 |
| 9 | [集成学习](09_ensemble_learning/) | AdaBoost、Bagging/OOB、随机森林 |
| 10 | [LDA](10_lda/) | 二分类 Fisher LDA |
| 11 | [SVM](11_svm/) | 线性 SMO |

每个入口都提供 `README.md`、`starter.py`、`demo.py`、`check.py` 和严格参考实现。日常练习只看输入、期望输出和实际输出；基础数组形状与数据内容校验暂不塞进这条算法主线。

每类算法先保留一个便于考场复写的代表实现。教材中的进阶变式和已验证扩展继续保留在 [advanced/](advanced/) 及原章节目录，不要求当前一次学完。

教材章节映射与完整仓库证据见 [docs/curriculum/](../docs/curriculum/)。
