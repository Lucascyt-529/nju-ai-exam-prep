# 贝叶斯分类器 6：EM 算法

当模型含未观测的隐变量时，直接最大化边际似然可能困难。EM 交替执行：

- E步：在当前参数下计算隐变量后验，即“责任度”；
- M步：把责任度当作软计数，最大化期望完整数据对数似然。

## 二项混合实验

每组实验观测 `heads[i] / tosses[i]`，但不知道它来自硬币0还是硬币1。参数包括：

```text
mixing.shape = (2,)       # 选择两枚硬币的概率
probabilities.shape = (2,) # 两枚硬币正面概率
responsibilities.shape = (n, 2)
```

E步在对数空间计算两个成分的后验并逐行归一化；M步使用责任度更新混合比例和正面概率。每轮记录观测数据对数似然，它理论上不应下降。

## Q函数、熵与证据下界

对任意逐行归一化的责任度`q(z)`：

```text
Q(theta | q) = sum_i,k q_ik * log p(x_i,z_i=k | theta)
H(q)          = -sum_i,k q_ik * log q_ik
ELBO(q,theta) = Q(theta | q) + H(q)
log p(X|theta) = ELBO(q,theta) + sum_i KL(q_i || p(z_i|x_i,theta))
```

因此ELBO不超过观测对数似然。E步把`q`设为当前参数下的真实隐变量后验，使KL为0、下界贴紧；M步固定`q`最大化Q，熵不变，所以抬高下界；下一次E步再把新参数处的下界贴紧。这给出单轮不等式链：

```text
old likelihood = old tight bound
               <= bound after M
               <= new likelihood = new tight bound after E
```

组合数项虽不影响M步最优参数，但参考实现把它纳入完整数据联合对数概率，使ELBO与观测似然的数值恒等式可以精确检查。`0*log(0)`按极限记为0。

EM只能保证到达局部驻点，成分标签还可整体交换；初始化仍然重要。本实验用于理解通用E/M控制流程，高斯混合聚类将在第9章单独实现。

## 学生任务

1. 手算一组观测属于两枚硬币的未归一化概率；
2. 用log-sum-exp得到责任度；
3. 验证每行责任度和为1；
4. 把责任度作为软计数完成M步；
5. 检查参数范围和似然历史；
6. 改变初始化，观察局部解与标签交换。
7. 对任意责任度验证`likelihood-ELBO=KL>=0`；
8. 固定一次E步责任度，比较M步前后的Q和ELBO；
9. 用`em_step_diagnostics`检查一轮EM的不等式链。
