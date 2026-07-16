# 第3章：线性模型

当前先建设线性回归。对数几率回归、线性判别分析和多分类策略将在后续批次加入。

## 当前结构

| 顺序 | 专题 | 状态 |
| --- | --- | --- |
| 1 | `01_linear_regression/`：预测、MSE、解析最小二乘与梯度下降 | 已建参考闭环 |
| 2 | `integrated_tasks/02_linear_regression_csv/`：三份CSV、验证指标、模型保存与测试预测 | 已建参考闭环 |
| 3 | `02_logistic_regression/`：稳定概率、交叉熵、L2梯度与梯度下降 | 已建参考闭环 |
| 4 | 线性判别分析 | 待建设 |
| 5 | 多分类学习 | 待建设 |

## 统一形状约定

```text
X.shape == (n_samples, n_features)
y.shape == (n_samples,)
w.shape == (n_features,)
prediction.shape == (n_samples,)
b 是标量
```

所有公式和代码都先写形状，再写运算。特别禁止把 `(n, 1)` 的预测与 `(n,)` 的标签直接相减。
