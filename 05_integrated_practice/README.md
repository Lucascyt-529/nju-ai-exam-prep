# 综合练习

综合题把文件读取、数据处理、模型训练、验证选择、模型保存和结果输出组合成一个无 OJ 闭环。它们不是新的孤立算法，而是检验基础能力能否真正协同工作。

## 当前任务

| 顺序 | 任务 | 组合能力 | 启用条件 |
| --- | --- | --- | --- |
| 1 | [预处理流水线](../integrated_tasks/01_preprocessing_pipeline/README.md) | 混合 CSV、填补、标准化、参数保存与恢复 | 完成基础数据处理 |
| 2 | [线性回归 CSV](../integrated_tasks/02_linear_regression_csv/README.md) | 三份数据、最小二乘、验证指标、模型恢复和预测 | 当前任务 |
| 3 | [逻辑回归 CSV](../integrated_tasks/03_logistic_regression_csv/README.md) | 标准化、分类指标/AUC、概率与标签 | 完成逻辑回归 |
| 4 | [K-means CSV](../integrated_tasks/04_kmeans_csv/README.md) | 缺失值、多次初始化、轮廓系数选 `k` | 完成 K-means |
| 5 | [混合特征分类](../integrated_tasks/05_mixed_feature_classification/README.md) | 训练词表、未知桶、JSON/NPZ 状态和独立预测 | 完成编码与分类主线 |

所有任务的参考实现和严格测试已经建成，但学习状态只在独立完成后更新。

## 三层练习方式

### Guided

允许查看专题 README、函数职责和输入输出契约。适合第一次把多个模块串起来。

### Independent

只根据题目、数据文件和输出要求从空文件实现。函数如何拆分由学习者自己决定。

### Transfer

改变至少一个条件：数据列、模型、评价指标、保存格式或命令行接口，验证是否真正掌握结构。

## 完整任务检查表

- 训练、验证、测试职责清楚；
- 预处理参数只从允许的数据拟合；
- 模型保存后重新加载再预测；
- 输出列名、顺序、小数位和换行符合要求；
- 同一随机种子可复现；
- 不通过查看测试集答案修改方案；
- 能解释每个函数的输入、输出和下游用途。

当前继续 [线性回归 CSV 任务](../integrated_tasks/02_linear_regression_csv/README.md)，本地学生代码不会被参考实现覆盖。
