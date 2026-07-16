# 综合任务2：线性回归训练、验证与测试预测

## 任务目标

在没有 OJ 调用函数的情况下，完成下面整条链路：

```text
读取 train.csv
→ NumPy 最小二乘拟合线性回归
→ 读取 validation.csv
→ 计算验证 MSE 与 R²
→ 保存并重新加载模型 .npz
→ 读取 test.csv 并预测
→ 严格保存 predictions.csv 和 metrics.txt
```

核心实现只能使用 Python 标准库和 NumPy。

## 数据格式

训练集和验证集：

```text
sample_id,feature_1,feature_2,target
```

测试集：

```text
sample_id,feature_1,feature_2
```

本题不含缺失值；缺失值预处理已在综合任务1单独训练。三个文件中的样本编号在各自文件内必须唯一。

## 输出格式

预测文件：

```text
sample_id,prediction
P01,10.000000
```

指标文件：

```text
validation_mse=0.000000
validation_r2=1.000000
```

模型文件必须且只包含 `weights` 和 `bias`，并能够通过 `allow_pickle=False` 加载。

## 运行前预测

样例数据满足：

```text
target = 2 * feature_1 + 3 * feature_2 + 1
```

先手算验证集和测试集的结果，再运行参考程序。特别说明：验证集只能评估，不能参与拟合参数。

## 参考运行

```bash
python integrated_tasks/02_linear_regression_csv/reference/solution.py --train integrated_tasks/02_linear_regression_csv/data/train.csv --validation integrated_tasks/02_linear_regression_csv/data/validation.csv --test integrated_tasks/02_linear_regression_csv/data/test.csv --predictions predictions.csv --metrics metrics.txt --model linear_model.npz
```

## 无 OJ 自测

- [ ] `X` 始终二维，`y` 与预测始终一维；
- [ ] 手算样例的 `w=[2,3]`、`b=1`；
- [ ] 修改验证集目标不会改变已拟合模型；
- [ ] 单行测试集仍输出一行预测；
- [ ] 错误表头、重复编号、非有限值被拒绝；
- [ ] 模型保存后在新进程中恢复；
- [ ] 两个输出文件逐字节符合格式。

## 迁移题

1. 增加第三个特征；
2. 在训练目标中加入小噪声，解释验证指标为什么不再完美；
3. 使用梯度下降替换最小二乘，对比参数和预测；
4. 加入综合任务1的训练集预处理，但验证和测试只能复用训练参数；
5. 增加命令行参数，在是否拟合截距之间切换。
