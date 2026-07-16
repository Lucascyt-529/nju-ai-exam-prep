# 《机器学习》覆盖矩阵

当前矩阵依据建仓任务书中的16章主线建立。它是初始映射，不等同于已经覆盖；待确认教材版次后，还需要按实际目录细化到小节。

状态只能使用：`未开始`、`已建骨架`、`已讲解`、`已实现待验证`、`已验证`、`学生已独立完成`。

| 教材章节 | 核心条目 | 讲解文件 | 实现文件 | 练习 | 测试 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 第1章 绪论 | 基本术语、假设空间、归纳偏好、NFL | — | — | — | — | 未开始 | 以概念辨析和小实验为主 |
| 第2章 模型评估与选择 | 经验误差、泛化误差、欠拟合与过拟合 | `watermelon_book/02_model_evaluation_selection/README.md` | — | — | — | 已建骨架 | 尚需过拟合实验与学生讲解 |
| 第2章 模型评估与选择 | 留出法、分层划分、K折与自助法 | `01_data_splitting/README.md` | `01_data_splitting/reference/solution.py` | `01_data_splitting/starter.py` | `test_model_evaluation_splitting_reference.py` | 已验证 | 路径相对第2章目录；含10项测试 |
| 第2章 模型评估与选择 | 回归、二分类、ROC/AUC与代价敏感度量 | `02_metrics/README.md` | `02_metrics/reference/solution.py` | `02_metrics/starter.py` | `test_model_evaluation_metrics_reference.py` | 已验证 | 含12项测试；未替代学生作答 |
| 第2章 模型评估与选择 | 交叉验证、折外预测与候选方案选择 | `03_cross_validation/README.md` | `03_cross_validation/reference/solution.py` | `03_cross_validation/starter.py` | `test_model_evaluation_cross_validation_reference.py` | 已验证 | 含7项测试；强调折内拟合隔离 |
| 第2章 模型评估与选择 | 比较检验 | — | — | — | — | 未开始 | 后续覆盖假设检验、McNemar、Friedman/Nemenyi等 |
| 第2章 模型评估与选择 | 偏差—方差分解 | — | — | — | — | 未开始 | 需要推导和可复现实验 |
| 第3章 线性模型 | 线性回归预测、MSE、梯度、最小二乘与梯度下降 | `watermelon_book/03_linear_models/01_linear_regression/README.md` | `01_linear_regression/reference/solution.py` | `01_linear_regression/starter.py` | `test_linear_regression_reference.py` | 已验证 | 路径相对第3章目录；含10项测试 |
| 第3章 线性模型 | 线性回归完整数据任务 | `integrated_tasks/02_linear_regression_csv/README.md` | `integrated_tasks/02_linear_regression_csv/reference/solution.py` | `integrated_tasks/02_linear_regression_csv/starter.py` | `test_integrated_linear_regression_reference.py` | 已验证 | 三份CSV、验证指标、模型恢复与严格输出；含8项测试 |
| 第3章 线性模型 | 对数几率回归、LDA、多分类 | — | — | — | — | 未开始 | NumPy 手写 |
| 第4章 决策树 | 划分选择与树生成 | — | — | — | — | 未开始 | Python/NumPy 手写 |
| 第4章 决策树 | 剪枝、连续值、缺失值、多变量树 | — | — | — | — | 未开始 | 分阶段实现 |
| 第5章 神经网络 | 感知机、多层网络与 BP | — | — | — | — | 未开始 | NumPy 手写反向传播 |
| 第5章 神经网络 | 全局最小、局部极小与其他常见网络 | — | — | — | — | 未开始 | 推导和小实验 |
| 第6章 支持向量机 | 间隔、对偶与核函数 | — | — | — | — | 未开始 | 核心版本手写 |
| 第6章 支持向量机 | 软间隔、正则化、支持向量回归 | — | — | — | — | 未开始 | 推导与代表实现 |
| 第7章 贝叶斯分类器 | 贝叶斯决策、极大似然、朴素贝叶斯 | — | — | — | — | 未开始 | Python/NumPy 手写 |
| 第7章 贝叶斯分类器 | 半朴素贝叶斯与贝叶斯网 | — | — | — | — | 未开始 | 代表方法和推断实验 |
| 第8章 集成学习 | AdaBoost、Bagging、随机森林 | — | — | — | — | 未开始 | 代表算法手写 |
| 第8章 集成学习 | 结合策略与多样性 | — | — | — | — | 未开始 | 性质实验 |
| 第9章 聚类 | 距离度量与性能度量 | — | — | — | — | 未开始 | NumPy 实现 |
| 第9章 聚类 | 原型、层次与密度聚类 | — | — | — | — | 未开始 | 代表算法手写 |
| 第10章 降维与度量学习 | KNN、MDS、PCA | — | — | — | — | 未开始 | 核心算法手写 |
| 第10章 降维与度量学习 | 核化、流形学习与度量学习 | — | — | — | — | 未开始 | 代表方法和实验 |
| 第11章 特征选择与稀疏学习 | 子集搜索、过滤式与包裹式选择 | — | — | — | — | 未开始 | 核心方法实现 |
| 第11章 特征选择与稀疏学习 | L1、字典学习与压缩感知 | — | — | — | — | 未开始 | 推导和实验 |
| 第12章 计算学习理论 | PAC 与有限假设空间 | — | — | — | — | 未开始 | 推导和概念题 |
| 第12章 计算学习理论 | VC维、Rademacher复杂度、稳定性 | — | — | — | — | 未开始 | 推导和验证实验 |
| 第13章 半监督学习 | 生成式与半监督 SVM | — | — | — | — | 未开始 | 代表算法手写 |
| 第13章 半监督学习 | 图方法与基于分歧的方法 | — | — | — | — | 未开始 | 代表算法手写 |
| 第14章 概率图模型 | HMM、MRF、CRF | — | — | — | — | 未开始 | 小型模型与推断 |
| 第14章 概率图模型 | 学习与推断、LDA话题模型 | — | — | — | — | 未开始 | 代表实验 |
| 第15章 规则学习 | 序贯覆盖、剪枝与命题规则 | — | — | — | — | 未开始 | 代表算法手写 |
| 第15章 规则学习 | 关系规则学习 | — | — | — | — | 未开始 | 概念与代表实验 |
| 第16章 强化学习 | MDP、动态规划、蒙特卡洛 | — | — | — | — | 未开始 | Python/NumPy 手写 |
| 第16章 强化学习 | TD、Q-learning、SARSA、函数近似 | — | — | — | — | 未开始 | Python/NumPy 手写 |
