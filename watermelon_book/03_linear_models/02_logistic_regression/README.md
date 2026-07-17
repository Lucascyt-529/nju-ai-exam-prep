# 线性模型2：对数几率回归

## 从线性分数到概率

```text
logit = X @ w + b                    # (n,)
probability = sigmoid(logit)          # (n,)
```

sigmoid把任意实数映射到 `(0,1)`。对数几率满足：

```text
log(p / (1-p)) = X @ w + b
```

这也是“对数几率回归”名称的来源。虽然常被称为logistic regression，它解决的是分类问题。

## 数值稳定

直接计算 `exp(-z)` 在很大的负 `z` 上可能溢出。稳定sigmoid按正负分支计算。

交叉熵也不先计算概率再做 `log(p)`，而是直接从logit计算：

```python
logaddexp(0, logit) - y * logit
```

这样在 `logit=1000` 或 `-1000` 时仍得到有限损失。

## 损失、L2与梯度

本专题定义：

```text
loss = mean(binary_cross_entropy) + 0.5 * l2 * sum(w²)
```

不正则化截距 `b`。梯度为：

```text
error = sigmoid(X @ w + b) - y       # (n,)
gradient_w = X.T @ error / n + l2*w  # (d,)
gradient_b = mean(error)              # 标量
```

## Hessian与牛顿法

把截距并入增广设计矩阵：

```text
design = [X, 1]                       # (n,d+1)
curvature = p * (1-p)                # (n,)
H = design.T @ (design * curvature[:,None]) / n
```

参数统一按 `[w_1,...,w_d,b]` 排列，因此Hessian形状是 `(d+1,d+1)`。L2只加
到前 `d` 个权重对角项，不正则化截距。牛顿方向通过线性方程求解：

```text
(H + damping*I) @ direction = gradient
beta_new = beta - step_size * direction
```

数值阻尼 `damping` 与模型L2不是一回事：L2改变训练目标且只作用于权重；阻尼
用于让当前Hessian更容易求解，这里加到全部参数方向。参考训练还使用回溯步长，
只接受非增损失。`damping=0` 遇到重复特征或退化设计时会明确报告Hessian奇异。

## 决策阈值

默认 `probability >= 0.5` 预测为1，但阈值不是模型训练参数。类别不均衡或错误代价不同时，应在验证集上结合precision、recall、ROC或代价选择阈值，不能用测试集调阈值。

## 运行前预测

1. `sigmoid(0)` 是多少；
2. logit很大为正或为负时，概率趋向哪里；
3. `X.T @ error` 为什么与 `w` 形状一致；
4. 增大L2为什么会推动权重靠近0，但不直接推动截距。

## 学生任务和迁移

1. 完成 `starter.py`；
2. 用有限差分验证一个权重和截距梯度；
3. 把阈值从0.5改为0.3，比较precision与recall；
4. 构造极端logit，证明损失没有NaN或无穷大；
5. 比较不同L2强度下权重范数和验证集表现。
6. 用有限差分检查Hessian的每一列是否等于梯度变化率；
7. 手算一次`H @ direction = gradient`并核对牛顿更新符号；
8. 比较若干次牛顿迭代与更多次梯度下降的损失，但同时记录每步计算/存储开销；
9. 复制一列特征制造秩亏，分别用`damping=0`和正阻尼运行；
10. 解释为什么牛顿法局部收敛快不等于所有数据规模下都更合适。
