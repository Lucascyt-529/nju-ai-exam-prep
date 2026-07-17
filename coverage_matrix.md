# 《机器学习》覆盖矩阵

当前矩阵依据建仓任务书中的16章主线建立。它是初始映射，不等同于已经覆盖；待确认教材版次后，还需要按实际目录细化到小节。当前建设优先级是第1～10章，第11～16章暂缓并保留为长期范围。

状态只能使用：`未开始`、`已建骨架`、`已讲解`、`已实现待验证`、`已验证`、`学生已独立完成`。

| 教材章节 | 核心条目 | 讲解文件 | 实现文件 | 练习 | 测试 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 第1章 绪论 | 基本术语、假设空间、归纳偏好、NFL | — | — | — | — | 未开始 | 以概念辨析和小实验为主 |
| 第2章 模型评估与选择 | 经验误差、泛化误差、欠拟合与过拟合 | `04_overfitting_bias_variance/README.md` | `04_overfitting_bias_variance/reference/solution.py` | `04_overfitting_bias_variance/starter.py` | `test_overfitting_bias_variance_reference.py` | 已验证 | 多项式复杂度与学习曲线；未替代学生作答 |
| 第2章 模型评估与选择 | 留出法、分层划分、K折与自助法 | `01_data_splitting/README.md` | `01_data_splitting/reference/solution.py` | `01_data_splitting/starter.py` | `test_model_evaluation_splitting_reference.py` | 已验证 | 路径相对第2章目录；含10项测试 |
| 第2章 模型评估与选择 | 回归、二分类、ROC/AUC与代价敏感度量 | `02_metrics/README.md` | `02_metrics/reference/solution.py` | `02_metrics/starter.py` | `test_model_evaluation_metrics_reference.py` | 已验证 | 含12项测试；未替代学生作答 |
| 第2章 模型评估与选择 | 交叉验证、折外预测与候选方案选择 | `03_cross_validation/README.md` | `03_cross_validation/reference/solution.py` | `03_cross_validation/starter.py` | `test_model_evaluation_cross_validation_reference.py` | 已验证 | 含7项测试；强调折内拟合隔离 |
| 第2章 模型评估与选择 | 比较检验 | `05_comparison_tests/README.md` | `05_comparison_tests/reference/solution.py` | `05_comparison_tests/starter.py` | `test_model_comparison_reference.py` | 已验证 | 二项、配对/修正t统计量、McNemar、Friedman/Nemenyi；含14项测试 |
| 第2章 模型评估与选择 | 偏差—方差分解 | `04_overfitting_bias_variance/README.md` | `04_overfitting_bias_variance/reference/solution.py` | `04_overfitting_bias_variance/starter.py` | `test_overfitting_bias_variance_reference.py` | 已验证 | 重复采样、逐点偏差平方与方差；共享10项测试 |
| 第3章 线性模型 | 线性回归预测、MSE、梯度、最小二乘与梯度下降 | `watermelon_book/03_linear_models/01_linear_regression/README.md` | `01_linear_regression/reference/solution.py` | `01_linear_regression/starter.py` | `test_linear_regression_reference.py` | 已验证 | 路径相对第3章目录；含10项测试 |
| 第3章 线性模型 | 线性回归完整数据任务 | `integrated_tasks/02_linear_regression_csv/README.md` | `integrated_tasks/02_linear_regression_csv/reference/solution.py` | `integrated_tasks/02_linear_regression_csv/starter.py` | `test_integrated_linear_regression_reference.py` | 已验证 | 三份CSV、验证指标、模型恢复与严格输出；含8项测试 |
| 第3章 线性模型 | 对数几率回归 | `watermelon_book/03_linear_models/02_logistic_regression/README.md` | `02_logistic_regression/reference/solution.py` | `02_logistic_regression/starter.py` | `test_logistic_regression_reference.py` | 已验证 | 稳定sigmoid/logits交叉熵、L2梯度与梯度下降；含12项测试 |
| 第3章 线性模型 | 对数几率回归完整数据任务 | `integrated_tasks/03_logistic_regression_csv/README.md` | `integrated_tasks/03_logistic_regression_csv/reference/solution.py` | `integrated_tasks/03_logistic_regression_csv/starter.py` | `test_integrated_logistic_regression_reference.py` | 已验证 | 训练集标准化、验证多指标/AUC、模型恢复与概率输出；含11项测试 |
| 第3章 线性模型 | 二分类LDA、类内散度、Fisher投影与阈值判别 | `watermelon_book/03_linear_models/03_lda/README.md` | `03_lda/reference/solution.py` | `03_lda/starter.py` | `test_lda_reference.py` | 已验证 | 奇异散度使用伪逆；覆盖平移/投影缩放不变性与Fisher比率；含11项测试 |
| 第3章 线性模型 | 一对其余与一对一多分类策略 | `watermelon_book/03_linear_models/04_multiclass_reduction/README.md` | `04_multiclass_reduction/reference/solution.py` | `04_multiclass_reduction/starter.py` | `test_multiclass_reduction_reference.py` | 已验证 | 非连续标签、OvR分数、OvO投票、四类组合数与确定性平票；含15项测试 |
| 第3章 线性模型 | 类别不平衡、再缩放、采样与阈值移动 | `watermelon_book/03_linear_models/05_class_imbalance/README.md` | `05_class_imbalance/reference/solution.py` | `05_class_imbalance/starter.py` | `test_class_imbalance_reference.py` | 已验证 | 类别权重、欠/过采样索引、代价阈值、先验漂移概率修正；含17项测试 |
| 第4章 决策树 | 熵、信息增益、增益率、基尼指数与离散特征树 | `watermelon_book/04_decision_trees/01_discrete_tree/README.md` | `01_discrete_tree/reference/solution.py` | `01_discrete_tree/starter.py` | `test_discrete_decision_tree_reference.py` | 已验证 | 含增益率两步选择、XOR零增益递归、空分支与未见值回退；含21项测试 |
| 第4章 决策树 | 验证集预剪枝与自底向上后剪枝 | `watermelon_book/04_decision_trees/02_pruning/README.md` | `01_discrete_tree/reference/solution.py` | `02_pruning/starter.py` | `test_decision_tree_pruning_reference.py` | 已验证 | 训练集选特征、验证集决定剪枝；准确率相同时保守剪枝；含13项测试 |
| 第4章 决策树 | 连续阈值与离散/连续混合树 | `watermelon_book/04_decision_trees/03_continuous_mixed_tree/README.md` | `03_continuous_mixed_tree/reference/solution.py` | `03_continuous_mixed_tree/starter.py` | `test_continuous_mixed_tree_reference.py` | 已验证 | 相邻不同值中点、确定性阈值、连续特征重复使用与范围外预测；含17项测试 |
| 第4章 决策树 | 缺失值、带权增益与分支概率传播 | `watermelon_book/04_decision_trees/04_missing_values/README.md` | `04_missing_values/reference/solution.py` | `04_missing_values/starter.py` | `test_missing_value_tree_reference.py` | 已验证 | 非缺失比例惩罚、样本权重守恒与缺失预测概率汇总；含17项测试 |
| 第4章 决策树 | 多变量线性划分树 | `watermelon_book/04_decision_trees/05_multivariate_tree/README.md` | `05_multivariate_tree/reference/solution.py` | `05_multivariate_tree/starter.py` | `test_multivariate_tree_reference.py` | 已验证 | 投影阈值、确定性坐标搜索、斜边界与旋转数据；含16项测试 |
| 第5章 神经网络 | 神经元模型与感知机 | `watermelon_book/05_neural_networks/01_perceptron/README.md` | `01_perceptron/reference/solution.py` | `01_perceptron/starter.py` | `test_perceptron_reference.py` | 已验证 | 线性得分、阈值预测、逐样本更新、可分收敛与XOR反例；含25项测试，未替代学生作答 |
| 第5章 神经网络 | 单隐层网络前向传播与标签形状 | `watermelon_book/05_neural_networks/02_forward_propagation/README.md` | `02_forward_propagation/reference/solution.py` | `02_forward_propagation/starter.py` | `test_neural_network_forward_reference.py` | 已验证 | 稳定sigmoid、可复现初始化、前向缓存、显式 `(n,)->(n,1)` 与稳定logits交叉熵；未替代学生作答 |
| 第5章 神经网络 | BP反向传播与数值梯度检查 | `watermelon_book/05_neural_networks/03_backpropagation/README.md` | `03_backpropagation/reference/solution.py` | `03_backpropagation/starter.py` | `test_neural_network_backprop_reference.py` | 已验证 | 输出层/隐层解析梯度、严格缓存和逐参数中心有限差分；含22项测试，未替代学生作答 |
| 第5章 神经网络 | 完整BP训练与预测 | `watermelon_book/05_neural_networks/04_bp_training/README.md` | `04_bp_training/reference/solution.py` | `04_bp_training/starter.py` | `test_neural_network_training_reference.py` | 已验证 | 非原地梯度下降、初始/逐轮损失、列形状预测、固定配置XOR拟合；含25项测试，未替代学生作答 |
| 第5章 神经网络 | 全局最小、局部极小与初始化 | `watermelon_book/05_neural_networks/05_optimization_landscape/README.md` | `05_optimization_landscape/reference/solution.py` | `05_optimization_landscape/starter.py` | `test_optimization_landscape_reference.py` | 已验证 | 可穷尽驻点的一维双井、二阶分类、深浅极小与多初值吸引域；含22项测试，未替代学生作答 |
| 第5章 神经网络 | RBF网络 | `watermelon_book/05_neural_networks/06_rbf_network/README.md` | `06_rbf_network/reference/solution.py` | `06_rbf_network/starter.py` | `test_rbf_network_reference.py` | 已验证 | 高斯距离响应、显式中心/宽度、最小二乘或岭输出层、奇异设计矩阵；含18项测试，未替代学生作答 |
| 第5章 神经网络 | SOM自组织映射 | `watermelon_book/05_neural_networks/07_som/README.md` | `07_som/reference/solution.py` | `07_som/starter.py` | `test_som_reference.py` | 已验证 | BMU、Gaussian一维拓扑邻域、非原地原型更新、衰减训练与量化误差；含25项测试，未替代学生作答 |
| 第5章 神经网络 | ART1自适应共振 | `watermelon_book/05_neural_networks/08_art1/README.md` | `08_art1/reference/solution.py` | `08_art1/starter.py` | `test_art1_reference.py` | 已验证 | 二值选择分数、警戒检验、确定性重置搜索、交集原型与增量类别；含21项测试，未替代学生作答 |
| 第5章 神经网络 | Elman循环网络 | `watermelon_book/05_neural_networks/09_elman_network/README.md` | `09_elman_network/reference/solution.py` | `09_elman_network/starter.py` | `test_elman_network_reference.py` | 已验证 | 上下文隐状态、线性输出、整段/分段前向、状态续接与重置；含19项测试，未实现BPTT，未替代学生作答 |
| 第5章 神经网络 | 其他常见网络 | — | — | — | — | 未开始 | 待按实际教材目录核对后建设级联相关与Boltzmann等代表性原理和实验 |
| 第6章 支持向量机 | 超平面、函数间隔与几何间隔 | `watermelon_book/06_support_vector_machines/01_margin_geometry/README.md` | `01_margin_geometry/reference/solution.py` | `01_margin_geometry/starter.py` | `test_svm_margin_geometry_reference.py` | 已验证 | 得分、点到超平面距离、正比例缩放不变性、规范化和最小间隔点；含19项测试，未替代学生作答 |
| 第6章 支持向量机 | 线性对偶优化与SMO | `watermelon_book/06_support_vector_machines/02_linear_smo/README.md` | `02_linear_smo/reference/solution.py` | `02_linear_smo/starter.py` | `test_svm_linear_smo_reference.py` | 已验证 | Gram矩阵、盒/等式约束、确定性成对更新、退化端点、KKT与原始-对偶目标；含20项测试，未替代学生作答 |
| 第6章 支持向量机 | 核函数与非线性分类 | `watermelon_book/06_support_vector_machines/04_kernel_svm/README.md` | `04_kernel_svm/reference/solution.py` | `04_kernel_svm/starter.py` | `test_svm_kernel_reference.py` | 已验证 | 线性/多项式/RBF核、核化SMO与仅支持向量预测；含24项测试，未替代学生作答 |
| 第6章 支持向量机 | 软间隔、松弛变量、hinge损失与C | `watermelon_book/06_support_vector_machines/03_soft_margin/README.md` | `03_soft_margin/reference/solution.py` | `03_soft_margin/starter.py` | `test_svm_soft_margin_reference.py` | 已验证 | 样本间隔区域、alpha状态、KKT映射与多个C的目标分量实验；含24项测试，未替代学生作答 |
| 第6章 支持向量机 | 支持向量回归 | `watermelon_book/06_support_vector_machines/05_epsilon_svr/README.md` | `05_epsilon_svr/reference/solution.py` | `05_epsilon_svr/starter.py` | `test_svm_epsilon_svr_reference.py` | 已验证 | epsilon不敏感损失、合并双对偶系数、成对坐标优化与线性/RBF核预测；含24项测试，未替代学生作答 |
| 第7章 贝叶斯分类器 | 最小风险贝叶斯决策与代价矩阵 | `watermelon_book/07_bayesian_classifiers/01_bayes_decision/README.md` | `01_bayes_decision/reference/solution.py` | `01_bayes_decision/starter.py` | `test_bayes_decision_reference.py` | 已验证 | 后验归一化、0/1损失、一般条件风险与非对称二分类阈值；含28项测试，未替代学生作答 |
| 第7章 贝叶斯分类器 | 极大似然参数估计 | `watermelon_book/07_bayesian_classifiers/02_maximum_likelihood/README.md` | `02_maximum_likelihood/reference/solution.py` | `02_maximum_likelihood/starter.py` | `test_bayes_mle_reference.py` | 已验证 | 类先验、Bernoulli、类别分布、单/多元高斯MLE与对数似然；含17项测试，未替代学生作答 |
| 第7章 贝叶斯分类器 | 朴素贝叶斯 | `watermelon_book/07_bayesian_classifiers/03_naive_bayes/README.md` | `03_naive_bayes/reference/solution.py` | `03_naive_bayes/starter.py` | `test_naive_bayes_reference.py` | 已验证 | 离散Laplace/未知桶、对数概率、高斯方差下限与稳定后验；含21项测试，未替代学生作答 |
| 第7章 贝叶斯分类器 | AODE半朴素贝叶斯 | `watermelon_book/07_bayesian_classifiers/04_aode/README.md` | `04_aode/reference/solution.py` | `04_aode/starter.py` | `test_aode_reference.py` | 已验证 | 超父频数阈值、SPODE对数分数、平均汇总与朴素回退；含19项测试，未替代学生作答 |
| 第7章 贝叶斯分类器 | 贝叶斯网与推断 | `watermelon_book/07_bayesian_classifiers/05_bayesian_network/README.md` | `05_bayesian_network/reference/solution.py` | `05_bayesian_network/starter.py` | `test_bayesian_network_reference.py` | 已验证 | 二值DAG/CPT校验、联合分解、证据求和与枚举精确后验；含14项测试，未替代学生作答 |
| 第8章 集成学习 | AdaBoost与加权决策树桩 | `watermelon_book/08_ensemble_learning/01_adaboost/README.md` | `01_adaboost/reference/solution.py` | `01_adaboost/starter.py` | `test_adaboost_reference.py` | 已验证 | 加权错误、分类器权重、样本权重更新、树桩枚举与强分类器；含20项测试，未替代学生作答 |
| 第8章 集成学习 | Bootstrap、Bagging与OOB评估 | `watermelon_book/08_ensemble_learning/02_bagging_oob/README.md` | `02_bagging_oob/reference/solution.py` | `02_bagging_oob/starter.py` | `test_bagging_oob_reference.py` | 已验证 | 有放回索引、树桩Bagging、硬投票、OOB覆盖与准确率；含19项测试，未替代学生作答 |
| 第8章 集成学习 | 随机森林 | `watermelon_book/08_ensemble_learning/03_random_forest/README.md` | `03_random_forest/reference/solution.py` | `03_random_forest/starter.py` | `test_random_forest_reference.py` | 已验证 | Bootstrap、随机候选特征、受限树桩、投票与预测相关性；含19项测试，未替代学生作答 |
| 第8章 集成学习 | 结合策略与多样性 | `watermelon_book/08_ensemble_learning/04_combination_diversity/README.md` | `04_combination_diversity/reference/solution.py` | `04_combination_diversity/starter.py` | `test_ensemble_combination_diversity_reference.py` | 已验证 | 加权平均、硬/软投票、Q/相关/分歧/双错与回归误差-分歧；含19项测试，未替代学生作答 |
| 第9章 聚类 | 距离度量与性能度量 | `watermelon_book/09_clustering/01_distances_metrics/README.md` | `01_distances_metrics/reference/solution.py` | `01_distances_metrics/starter.py` | `test_clustering_distances_metrics_reference.py` | 已验证 | Minkowski/Hamming、SSE、轮廓、样本对计数、Rand与ARI；含21项测试，未替代学生作答 |
| 第9章 聚类 | K-means原型聚类 | `watermelon_book/09_clustering/02_kmeans/README.md` | `02_kmeans/reference/solution.py` | `02_kmeans/starter.py` | `test_kmeans_reference.py` | 已验证 | K-means++、分配/更新、空簇重置、SSE历史与收敛；含23项测试，未替代学生作答 |
| 第9章 聚类 | LVQ监督原型学习 | `watermelon_book/09_clustering/03_lvq/README.md` | `03_lvq/reference/solution.py` | `03_lvq/starter.py` | `test_lvq_reference.py` | 已验证 | 分层原型初始化、最近原型、正确靠近/错误远离、衰减训练与预测；含18项测试，未替代学生作答 |
| 第9章 聚类 | AGNES层次聚类 | `watermelon_book/09_clustering/04_agnes/README.md` | `04_agnes/reference/solution.py` | `04_agnes/starter.py` | `test_agnes_reference.py` | 已验证 | 单/全/平均链接、确定性合并、成员/距离历史与目标簇标签；含25项测试，未替代学生作答 |
| 第9章 聚类 | DBSCAN密度聚类 | — | — | — | — | 未开始 | 核心点、密度可达与噪声 |
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
