# 逻辑回归

逻辑回归用于二分类。它先像线性回归一样计算线性得分，再通过 sigmoid 把任意实数压到0和1之间，解释为属于正类的概率。

## 算法概览

逻辑回归虽然名称中有“回归”，主要用于分类。它假设正类概率的对数几率可以由特征的线性组合表示：线性得分越大，属于正类的概率越高。这样既保留了线性模型清晰的决策边界，又能输出连续概率，而不仅是一个硬标签。

从广义线性模型角度看，它使用对数几率作为联系函数，把0到1之间的概率映射到整个实数轴。模型通常通过极大似然估计参数，对应到代码中就是最小化交叉熵损失。

逻辑回归适合需要概率、阈值可调和一定可解释性的二分类问题。它的边界仍是线性的，因此面对无法用一个超平面分开的数据时，需要特征变换、核方法或非线性模型。

## 输入、输出与参数

```text
logit = X @ weights + bias
probability = sigmoid(logit)
label = 1 if probability >= threshold else 0
```

- `X`：每行一个样本，每列一个数值特征；
- `y`：只包含0和1的真实标签；
- `weights`：每个特征的权重；
- `bias`：截距；
- `probability`：模型给出的正类概率；
- `threshold`：把概率转为类别的决策阈值，默认通常为0.5。

权重为正表示该特征增大会提高正类的 logit，权重为负则相反。逻辑回归学到的决策边界仍然是线性的，但输出不再是任意实数。

## 核心公式

Sigmoid：

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

单个样本的二元交叉熵：

```text
loss = -y * log(p) - (1-y) * log(1-p)
```

直接从 logit 计算更稳定：

```text
loss = log(1 + exp(logit)) - y * logit
```

对全部样本取平均后，可再加 L2 正则项：

```text
total_loss = mean(data_loss) + 0.5 * l2 * (weights @ weights)
```

截距通常不加入 L2。梯度为：

```text
error = probability - y
gradient_weights = X.T @ error / n + l2 * weights
gradient_bias = mean(error)
```

## 完整算法流程

```text
初始化 weights, bias
-> 计算 logits
-> 计算 probabilities
-> 计算交叉熵损失
-> 计算梯度
-> 更新 weights, bias
-> 重复训练
-> 在验证集上比较概率和分类指标
```

Hessian、牛顿方向和 `fit_newton` 是第二层内容。先完整掌握梯度下降，再比较一阶与二阶方法。

## 函数职责与实现顺序

| 函数 | 输入 | 输出 | 下游用途 |
| --- | --- | --- | --- |
| `stable_sigmoid` | 任意 logits | 同位置概率 | 概率预测、梯度 |
| `binary_cross_entropy_from_logits` | `y, logits` | 平均损失 | 稳定评价当前参数 |
| `predict_proba` | `X, weights, bias` | 每个样本的概率 | 损失、预测、指标 |
| `logistic_loss` | 数据、参数、L2 | 总损失 | 训练历史、模型比较 |
| `logistic_gradients` | 数据、参数、L2 | 权重梯度、截距梯度 | 梯度下降 |
| `fit_gradient_descent` | 训练数据和训练设置 | 参数与损失历史 | 完整训练 |
| `predict_labels` | 概率和阈值 | 0/1标签 | 分类指标 |

```text
stable_sigmoid
-> predict_proba
-> binary_cross_entropy_from_logits
-> logistic_loss
-> logistic_gradients
-> fit_gradient_descent
-> predict_labels
```

## 最小手算样例

若：

```text
X = [[1, 2]]
weights = [0.5, -1]
bias = 0.5
```

则：

```text
logit = 1*0.5 + 2*(-1) + 0.5 = -1
probability = sigmoid(-1) ≈ 0.2689
```

阈值为0.5时预测标签为0。若真实标签 `y=1`，当前模型对正确类别信心较低，交叉熵会较大；若 `y=0`，损失会较小。

`check.py` 先核对：

```text
sigmoid([-2, 0, 2]) ≈ [0.119203, 0.5, 0.880797]
```

## 如何评价

训练损失下降只说明优化过程在当前训练目标上前进。验证阶段还应查看：

- accuracy：全部样本中预测正确的比例；
- precision：预测为正的样本中有多少是真的正类；
- recall：真实正类中找回了多少；
- F1：precision 与 recall 的调和平均；
- AUC：不固定单一阈值时的排序能力。

类别不平衡时只看 accuracy 很危险。阈值也属于决策方案，应在验证集上选择，不能使用测试集调整。

## 运行与核对

```powershell
python 02_machine_learning/02_logistic_regression/demo.py
python 02_machine_learning/02_logistic_regression/check.py
```

完整 CSV 闭环见 [逻辑回归综合任务](../../integrated_tasks/03_logistic_regression_csv/README.md)。

## 常见错误

- 直接对极大负数计算 `exp(-z)` 导致溢出；
- 把逻辑回归损失误写成 MSE，却仍套用交叉熵梯度；
- 梯度中的 `error` 写成 `y - probability`，更新方向随之反转；
- L2 错误地正则化截距；
- `predict_proba` 直接返回0/1标签，丢失概率信息；
- 用分类标签而不是连续概率计算 AUC；
- 在测试集上选择阈值。

## 自学闭环

1. 手算一个样本的 logit、概率和标签；
2. 完成梯度下降主线并确认损失总体下降；
3. 构造极端 logits，确认 sigmoid 仍返回有限值；
4. 改变阈值，解释 precision 与 recall 为什么会变化；
5. 比较无正则和较强 L2 下的权重大小及验证指标；
6. 最后再进入 Hessian 和牛顿法，并说明它们与梯度下降的区别。

完成这些后，再查看 [reference/solution.py](reference/solution.py) 核对，而不是先背参考实现。
