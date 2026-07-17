# 降维与度量学习 3：PCA

主成分分析（PCA）寻找数据方差最大的正交方向。它既可以解释为最大化投影方差，也可以解释为最小化线性重构误差。

## 先看形状

若 `X.shape == (n_samples, n_features)`，保留 `q` 个主成分：

```text
mean.shape       == (n_features,)
centered.shape   == (n_samples, n_features)
covariance.shape == (n_features, n_features)
components.shape == (q, n_features)
Z.shape          == (n_samples, q)
```

```python
mean = X.mean(axis=0)
centered = X - mean
Z = centered @ components.T
```

这里 `axis=0` 消去样本这一维，对每个特征求均值；得到的是一维数组 `(n_features,)`，它依靠广播从每一行中减去。

## 协方差与主轴

参考实现使用样本协方差：

```text
covariance = centered.T @ centered / (n_samples - 1)
```

对称矩阵特征分解后，按特征值从大到小选择主轴。第 `i` 个特征值表示第 `i` 个主成分承载的样本方差；解释方差比是它占全部特征值之和的比例。

当只有一个样本时，没有可估计的样本方差，参考实现把协方差设为零矩阵。常数数据的解释方差比也定义为全0，避免除以0。

## 变换与重构

```text
transform:         Z = (X - mean) @ components.T
inverse_transform: X_hat = Z @ components + mean
```

若保留全部主成分，重构应接近原数据；减少主成分会丢失较小方差方向的信息。

## 学生任务

1. 对一个 `3 x 2` 数据集按列手算均值；
2. 写出中心化矩阵并检查每列均值为0；
3. 手算 `2 x 2` 协方差矩阵；
4. 解释为什么主成分矩阵是 `(q, d)`；
5. 完成投影和逆变换并逐项核对形状；
6. 比较保留1维与全部维度的重构误差；
7. 对平移后的数据重复PCA，解释主轴和方差为什么不变。

先运行 `guided_demo.py`，再补全 `starter.py`。参考实现通过只说明仓库代码正确，不说明学习者已经会写。
