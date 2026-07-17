# 集成学习1：AdaBoost与决策树桩

AdaBoost顺序训练弱分类器。第 `t` 轮使用当前样本权重拟合分类器 `h_t`：

```text
error_t = sum_i weight_i * [h_t(x_i) != y_i]
alpha_t = 0.5 * log((1-error_t)/error_t)
weight_i <- weight_i * exp(-alpha_t*y_i*h_t(x_i))
```

更新后再归一化。正确样本乘 `exp(-alpha)`，错误样本乘 `exp(+alpha)`，所以错分样本相对权重上升。

## 决策树桩

本专题的弱分类器只选择一个特征和一个阈值：

```text
polarity=+1: x_j >= threshold -> +1，否则-1
polarity=-1: 上述预测整体取反
```

候选阈值包括相邻不同值中点和两端常数分类器。加权错误相同依次选择较小特征下标、较小阈值、`+1`极性，保证可复现。

## 强分类器

```text
F(x) = sum_t alpha_t*h_t(x)
H(x) = +1 if F(x)>=0 else -1
```

若树桩完美分类，使用数值下限计算有限alpha后提前停止；若最佳树桩不优于随机猜测，则不加入无效学习器。

## 形状

```text
X:(n,d) y:(n,) sample_weight:(n,) scores:(n,)
```

## 学生任务

1. 手算一轮加权错误率；
2. 手算alpha；
3. 比较更新前后错分样本相对权重；
4. 枚举一个特征的所有中点阈值；
5. 验证树桩平局规则；
6. 手算两个弱分类器的加权投票；
7. 解释训练错误下降不等于测试错误必然下降。

参考实现通过不等于学习者已经独立掌握。
