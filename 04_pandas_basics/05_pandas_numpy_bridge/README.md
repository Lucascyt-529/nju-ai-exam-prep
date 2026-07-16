# pandas 5：安全转换到 NumPy

## 先选列，再转换

机器学习程序不能直接对整张 DataFrame 调用 `to_numpy()`，因为表中可能包含编号、姓名或顺序变化。应显式写出：

```python
feature_columns = ["study_hours", "attendance"]
X = frame.loc[:, feature_columns].to_numpy(dtype=float)
y = frame.loc[:, "label"].to_numpy(dtype=int)
```

得到：

```text
X.shape == (n, d)
y.shape == (n,)
```

`frame[["label"]].to_numpy()` 会得到 `(n,1)`，不能在不检查的情况下与 `(n,)` 预测相减，否则可能广播成 `(n,n)`。

## 列顺序就是特征含义

`["study_hours", "attendance"]` 与反向顺序会产生不同矩阵。训练阶段保存的均值和尺度使用带列名的 Series；转换时必须与请求的特征列顺序完全一致。

## pandas 与 NumPy 的标准差差异

```text
pandas Series.std() 默认 ddof=1
NumPy ndarray.std() 默认 ddof=0
```

本仓库标准化统一使用总体标准差 `ddof=0`，因此 pandas 拟合结果应与：

```python
X_train.mean(axis=0)
X_train.std(axis=0)
```

一致。常数列尺度替换为1。

## 训练统计量隔离

`fit_table_standardizer` 只能接收训练表。验证表与测试表调用 `transform_features`，不得重新计算自己的均值或尺度。

## 学生任务和迁移

1. 预测单中括号与双中括号选择标签后的形状；
2. 交换两列特征顺序并解释矩阵变化；
3. 分别计算 pandas 默认标准差、`ddof=0` 和 NumPy 标准差；
4. 用训练表统计量转换数值范围完全不同的验证表；
5. 构造小数标签并尝试整数标签模式，解释为何拒绝；
6. 构造 `(n,1)` 预测，修正为 `(n,)` 后再保存。
