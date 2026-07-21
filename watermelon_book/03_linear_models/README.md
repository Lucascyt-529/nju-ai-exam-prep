# 第3章：线性模型

本章已按参考实现闭环覆盖线性回归、对数几率回归、线性判别分析、
多分类拆分与类别不平衡。这里的“已建”只表示仓库已有讲解、参考、学生入口
和自动测试，不表示学习者已经掌握；真实学习进度另见根目录 [LEARNING_STATUS.md](../../LEARNING_STATUS.md)。

## 当前结构

| 顺序 | 专题 | 状态 |
| --- | --- | --- |
| 1 | [新线性回归入口](../../02_machine_learning/01_linear_regression/README.md)：普通最小二乘与梯度下降；旧 `01_linear_regression/generalized_linear_models/` 暂留联系函数内容 | 普通线性回归已迁移 |
| 2 | `integrated_tasks/02_linear_regression_csv/`：三份CSV、验证指标、模型保存与测试预测 | 已建参考闭环 |
| 3 | [新逻辑回归入口](../../02_machine_learning/02_logistic_regression/README.md)：稳定概率、交叉熵、梯度/Hessian、梯度下降与牛顿法 | 已迁移 |
| 4 | [新LDA入口](../../02_machine_learning/10_lda/README.md)：二分类Fisher方向、多分类散度、K-1维监督投影与预测 | 已迁移 |
| 5 | `04_multiclass_reduction/`：OvR、OvO、MvM/ECOC编码、距离解码与纠错边界 | 已建参考闭环 |
| 6 | `05_class_imbalance/`：采样、类别权重、代价阈值与先验修正 | 已建参考闭环 |

## 统一形状约定

```text
X.shape == (n_samples, n_features)
y.shape == (n_samples,)
w.shape == (n_features,)
prediction.shape == (n_samples,)
b 是标量
```

所有公式和代码都先写形状，再写运算。特别禁止把 `(n, 1)` 的预测与 `(n,)` 的标签直接相减。
