# 支持向量机5：epsilon-SVR

## epsilon不敏感带

SVR不要求每个预测都恰好等于目标。误差落在宽度为 `2*epsilon` 的管道内时损失为0：

```text
residual_i = y_i - f(x_i)
loss_i = max(0, |residual_i| - epsilon)
```

因此，误差不超过epsilon的样本不会推动模型继续贴近它；触及或越过管道边界的样本才可能成为支持向量。

## 双对偶变量为什么可合并

标准SVR为管道上、下两侧分别引入 `alpha` 与 `alpha_hat`。预测只使用差：

```text
beta_i = alpha_i - alpha_hat_i
f(x) = sum_i beta_i K(x_i,x) + b
```

对偶问题可以写成：

```text
maximize  -0.5*beta^T K beta - epsilon*sum(|beta_i|) + y^T beta
subject to -C <= beta_i <= C
           sum(beta_i) = 0
```

本专题每次同时更新两个beta，以保持系数和为0；在由绝对值产生的分段点、区间端点和各光滑区间驻点中，直接选取对偶目标最大的候选。

## beta与残差的KKT对应

令 `r=y-f(x)`：

```text
beta = 0        -> |r| <= epsilon
0 < beta < C    -> r = +epsilon
beta = C        -> r >= +epsilon
-C < beta < 0   -> r = -epsilon
beta = -C       -> r <= -epsilon
```

自由支持向量因此可以恢复偏置。若没有自由支持向量，参考实现从KKT上下界选择可行偏置。

## 形状

```text
X_train: (n,d)    y: (n,)      beta: (n,)
K_query: (m,n)    prediction: (m,)
```

标签必须是一维回归目标，不能传 `(n,1)` 后依赖广播。

## 学生任务

1. 手算五个残差的epsilon损失；
2. 区分管道内、边界和管道外样本；
3. 解释为什么单独更新一个beta会破坏 `sum(beta)=0`；
4. 从自由正/负beta分别推导偏置公式；
5. 验证仅用支持向量预测与完整展开一致；
6. 比较不同epsilon下支持向量数量；
7. 用RBF核拟合弯曲数据，并说明这不等于自动选择了合适超参数。

参考实现通过不等于学习者已经独立掌握。
