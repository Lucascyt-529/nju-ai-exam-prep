# SVM

支持向量机寻找一个分开两类的超平面，并让离边界最近的训练样本到边界的间隔尽可能大。当前入口先实现线性软间隔 SVM 的对偶形式与简化 SMO。

## 算法概览

对于线性可分数据，能够正确分类训练样本的超平面往往不止一个。SVM 不任意选择其中一条，而是寻找具有最大间隔的边界，希望模型对训练样本的轻微扰动更稳健。

真正决定间隔的是离边界最近的少数样本，即支持向量。远离边界的样本通常不会影响最终超平面，这使 SVM 的解具有稀疏性。数据不可完全分开时，软间隔允许部分样本违反间隔要求，并用参数 `C` 平衡间隔大小和训练错误。

通过对偶形式，模型只依赖样本之间的内积。把普通内积替换为核函数，就能在不显式构造高维特征的情况下形成非线性边界。当前先掌握线性核和对偶优化，再进入核技巧。

## 线性决策函数

标签使用 `-1/1`：

```text
score = X @ weights + bias
prediction = 1 if score >= 0 else -1
```

决策边界为 `w.T @ x + b = 0`。规范化后几何间隔与 `1 / ||w||` 有关，因此最小化权重范数对应更大的间隔。

## 软间隔原始目标

允许少量样本进入间隔或被错分：

```text
0.5 * ||w||^2 + C * sum(hinge_loss_i)
hinge_loss_i = max(0, 1 - y_i * score_i)
```

- `C` 大：更重视训练错误，边界可能更贴合训练数据；
- `C` 小：更重视大间隔，允许更多训练违约；
- `C` 应在验证集上选择。

## 为什么学习对偶问题

对偶变量 `alpha_i` 对应每个训练样本。线性核 Gram 矩阵为：

```text
K = X @ X.T
```

对偶最大化目标：

```text
sum(alpha_i)
- 0.5 * sum_i sum_j alpha_i alpha_j y_i y_j K_ij
```

约束：

```text
0 <= alpha_i <= C
sum(alpha_i * y_i) = 0
```

只有 `alpha_i > 0` 的样本会参与最终决策，它们是支持向量。线性权重可恢复为：

```text
weights = sum_i alpha_i * y_i * x_i
```

## SMO 的核心思想

由于等式约束，不能只独立修改一个 `alpha`。SMO 每次选择两个变量，在保持约束成立的情况下：

1. 根据当前模型检查 KKT 条件；
2. 选择一对 `alpha_i, alpha_j`；
3. 计算第二个变量的新值并裁剪到可行区间；
4. 由等式约束更新第一个变量；
5. 更新截距 `b`；
6. 重复直到多轮没有有效更新或达到迭代上限。

这个入口的重点是理解对偶约束、成对更新和停止条件，而不是背完整优化推导。

## 函数职责

| 函数 | 输入 | 输出 | 下游用途 |
| --- | --- | --- | --- |
| `linear_kernel_matrix` | 两组样本 | 两两内积矩阵 | 对偶训练与预测 |
| `dual_objective` | `alphas, y, gram` | 对偶目标值 | 观察优化 |
| `fit_linear_svm_smo` | 训练数据与优化设置 | alpha、支持向量、截距等模型 | 完整训练 |
| `decision_function` | 模型、新数据 | 连续决策分数 | 标签预测 |
| `linear_weights` | 线性 SVM 模型 | 原始空间权重 | 解释边界 |
| `kkt_residuals` | 模型、容差 | 每个样本违约程度 | 检查收敛 |

推荐顺序：

```text
linear_kernel_matrix
-> dual_objective
-> 理解两个 alpha 的可行更新
-> fit_linear_svm_smo
-> decision_function
-> linear_weights
-> kkt_residuals
```

## 最小样例

```text
X = [[-2], [-1], [1], [2]]
y = [-1, -1, 1, 1]
```

Gram 矩阵是外积：

```text
K = X @ X.T
  = [[ 4,  2, -2, -4],
     [ 2,  1, -1, -2],
     [-2, -1,  1,  2],
     [-4, -2,  2,  4]]
```

离边界最近的 `-1` 和 `1` 更可能成为支持向量；更远的 `-2` 和 `2` 不必拥有非零 alpha。最终边界应位于0附近，训练预测为 `[-1,-1,1,1]`。

## KKT 条件的直觉

记带符号间隔 `margin_i = y_i * score_i`：

```text
alpha_i = 0      时，样本应在间隔外或边界上：margin_i >= 1
0 < alpha_i < C  时，样本应恰在间隔边界：margin_i = 1
alpha_i = C      时，样本可以位于间隔内：margin_i <= 1
```

`kkt_residuals` 把这些条件偏离了多少变成可检查数值。训练预测全对并不代表对偶优化已经满足 KKT。

## 运行与核对

```powershell
python 02_machine_learning/11_svm/demo.py
python 02_machine_learning/11_svm/check.py
```

完成线性主线后，再进入 [软间隔损失](../../watermelon_book/06_support_vector_machines/03_soft_margin/README.md)、[核 SVM](../../watermelon_book/06_support_vector_machines/04_kernel_svm/README.md) 和 SVR。

## 常见错误

- Gram 矩阵错误写成逐元素乘法；
- 对偶目标二次项漏掉标签乘积 `y_i*y_j`；
- 更新一个 alpha 后破坏 `sum(alpha*y)=0`；
- 没有把 alpha 裁剪到 `[0,C]`；
- 截距更新与新旧 alpha 混用；
- 把所有训练样本都当作支持向量；
- 用训练准确率选择 `C`；
- 预测标签正确就认为 KKT 已满足；
- 在线性核实现中混淆对偶系数与原始权重。

## 自学闭环

1. 手算本页 Gram 矩阵；
2. 给定简单 alpha，计算一次对偶目标和等式约束；
3. 解释为什么必须同时更新两个 alpha；
4. 完成 SMO 后检查盒约束和 `sum(alpha*y)`；
5. 从 alpha 恢复线性权重，并与对偶决策分数比较；
6. 改变 `C`，比较支持向量数量、训练结果和验证指标；
7. 最后检查 KKT 残差，而不只看分类正确率。

能解释支持向量、`C`、对偶约束和成对更新之间的关系，才算掌握线性 SVM 主线。
