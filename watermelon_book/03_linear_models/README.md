# 第3章：线性模型

本章已按参考实现闭环覆盖线性回归、对数几率回归、线性判别分析、
多分类拆分与类别不平衡。这里的“已建”只表示仓库已有讲解、参考、学生入口
和自动测试，不表示学习者已经掌握；真实学习进度另见 `progress.md`。

## 当前结构

| 顺序 | 专题 | 状态 |
| --- | --- | --- |
| 1 | `01_linear_regression/`：普通最小二乘、梯度下降、联系函数与对数线性回归 | 已建参考闭环 |
| 2 | `integrated_tasks/02_linear_regression_csv/`：三份CSV、验证指标、模型保存与测试预测 | 已建参考闭环 |
| 3 | `02_logistic_regression/`：稳定概率、交叉熵、梯度/Hessian、梯度下降与牛顿法 | 已建参考闭环 |
| 4 | `03_lda/`：类内散度、Fisher投影与二分类判别 | 已建参考闭环 |
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
