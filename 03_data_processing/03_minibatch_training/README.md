# 数据处理3：mini-batch与可复现训练数据流

## 一个epoch到底是什么

若不丢弃尾批，一个epoch应让每个训练样本恰好出现一次。`n=7,batch_size=3`时批大小应为`3,3,1`。`drop_last=True`才会主动丢弃最后1个样本，而且必须明确知道哪些数据被丢了。

## 为什么先生成下标

先对`indices`打乱，再用同一组下标取`X`和`y`：

```python
indices = rng.permutation(n)
X_batch = X[indices]
y_batch = y[indices]
```

若分别打乱`X`和`y`，形状仍正确，程序也能运行，但样本与标签已经错位。这是最危险的“静默错误”之一。

## 随机数规则

- 在整个训练计划开始时创建一次`np.random.default_rng(seed)`；
- 每个epoch继续使用同一个RNG；
- 相同种子应复现完整训练计划；
- 不要在每个epoch内部重新用相同种子创建RNG，否则每轮顺序完全相同；
- 测试应检查同种子相等、不同种子通常不同，而不硬背某个随机序列。

本参考实现一次返回完整下标计划，便于初学者审计和测试。真实大数据训练可改为逐epoch生成器以减少内存，但不能因此省略覆盖、同步和可复现性检查。

## 形状主线

```text
X.shape             == (n_samples,n_features)
y.shape             == (n_samples,)
indices.shape       == (batch_size,) 或尾批大小
X_batch.shape       == (len(indices),n_features)
y_batch.shape       == (len(indices),)
weights.shape       == (n_features,)
gradient_weights    == (n_features,)
```

## 学习步骤

1. 先在`shuffle=False`时写出所有批下标；
2. 再打开shuffle，验证一个epoch排序后仍为`0..n-1`；
3. 用下标同时提取`X/y`，构造可手算的对应关系；
4. 实现一个batch的MSE梯度；
5. 用`batch_size=n`验证一次更新等于全批梯度下降；
6. 最后接入多epoch训练并记录每轮完整训练集损失。

运行演示：

```text
python 03_data_processing/03_minibatch_training/guided_demo.py
```

## 迁移题

1. 分别取`batch_size=1,n,2n`，写出批次数和形状；
2. 打开`drop_last`，计算每轮实际使用样本数；
3. 错误地分别打乱`X/y`，用断言构造能发现错位的反例；
4. 把线性回归替换为逻辑回归，只修改损失和梯度函数；
5. 增加验证集，解释为什么验证集不应被mini-batch训练循环更新。
