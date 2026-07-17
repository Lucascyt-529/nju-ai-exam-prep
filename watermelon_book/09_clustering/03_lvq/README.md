# 聚类3：LVQ监督原型学习

LVQ（Learning Vector Quantization）维护若干带类别标签的原型。对每个训练样本找到最近原型：

```text
原型标签正确: p <- p + eta*(x-p)   向样本靠近
原型标签错误: p <- p - eta*(x-p)   远离样本
```

与K-means不同，LVQ训练时使用真实类别标签；它放在原型聚类章节中是因为共享“以代表点刻画数据”的思想，不代表它是无监督算法。

## 初始化

参考实现为每个类别无放回选取 `prototypes_per_class` 个本类训练样本，固定随机种子。原型标签顺序按排序后的类别，再按抽样下标排序，便于复现。

## 训练顺序与学习率

每轮按固定样本顺序更新，不额外打乱，以便手算。第 `epoch` 轮学习率：

```text
eta_epoch = learning_rate * decay^epoch
```

`0 < eta <= 1`，`0 < decay <= 1`。历史保存初始原型及每轮结束后的完整副本。

## 形状

```text
X:(n,d) prototypes:(m,d) prototype_labels:(m,)
distance:(n,m) prediction:(n,)
history:(epochs+1,m,d)
```

## 学生任务

1. 手算最近原型；
2. 手算类别正确的靠近更新；
3. 手算类别错误的远离更新；
4. 比较一次更新前后赢家距离；
5. 解释为什么原型标签固定不变；
6. 比较LVQ与K-means是否使用y；
7. 观察学习率衰减对后期步长的影响。

参考实现通过不等于学习者已经独立掌握。
