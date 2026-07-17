# 003：布尔掩码不要多包一层，X和y必须共享行索引

## 发生场景

首次实现条件筛选时写成：

```python
masks = [matrix[:, column] >= threshold]
```

比较表达式本来已经产生形状`(n,)`的布尔数组；外层方括号把它变成了包含一个数组的列表，转换后的结构相当于`(1,n)`，不能对应矩阵的`n`行。

## 正确形状

```python
mask = matrix[:, column] >= threshold

mask.shape       # (n,)
matrix[mask]     # 保留mask为True的完整行
```

筛选、排序、打乱样本都遵守同一规则：先得到一组行选择信息，再同时应用到`X`和`y`。

```python
order = rng.permutation(len(X))
X_shuffled = X[order]
y_shuffled = y[order]
```

## 当前状态

学习者已能解释回归任务中分别随机打乱`X`和`y`会破坏样本—标签对应关系，并完成布尔筛选、任意目标列切分和固定种子同步排列。学生专项测试验证掩码是一维布尔数组、排列恰好包含每个下标一次、同种子可复现且输入不被修改。后续在训练/验证划分中继续复验。
