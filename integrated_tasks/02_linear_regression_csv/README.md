# 综合任务2：线性回归 CSV 完整闭环

本题把算法函数放回完整程序：读取三份 CSV，只用训练集拟合，在验证集评价，保存并重新加载模型，再预测测试集并保存结果。核心只使用标准库和 NumPy。

## 三份 CSV

训练集和验证集的列顺序固定为：

```text
sample_id,feature_1,feature_2,target
```

测试集为：

```text
sample_id,feature_1,feature_2
```

`sample_id` 在第0列，只用于最终对应样本；中间连续的数值列组成 `X`；有标签时最后一列组成一维 `y`。因此 `has_target=True` 时取 `row[1:-1]` 与 `row[-1]`，返回 `(sample_ids, X, y)`；`has_target=False` 时取 `row[1:]`，返回的 y 是 `None`。测试集没有 target，因为它模拟最终未知答案，不能参与训练或选模型。

## 数据职责

```text
train:      拟合 weights 和 bias
validation: 计算 MSE、R2，检查方案泛化表现
test:       方案确定后生成最终预测，不用于调参
```

样例满足 `target = 2*feature_1 + 3*feature_2 + 1`，可先手算 `weights=[2,3]`、`bias=1`。

## 函数调用图

```text
main
└─ run_pipeline
   ├─ load_regression_csv(train, has_target=True)
   ├─ load_regression_csv(validation, has_target=True)
   ├─ load_regression_csv(test, has_target=False)
   ├─ fit_least_squares(X_train, y_train)
   ├─ predict(X_validation, weights, bias)
   ├─ regression_metrics(y_validation, validation_prediction)
   ├─ save_model -> load_model
   ├─ predict(X_test, loaded_weights, loaded_bias)
   ├─ save_predictions
   └─ save_metrics
```

## 输入输出契约

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `load_regression_csv` | 路径、是否含 target | ID 列表、二维 X、一维 y 或 None |
| `fit_least_squares` | 二维 X、一维 y | 一维 weights、Python float bias |
| `predict` | X、weights、bias | 一维预测 |
| `regression_metrics` | 真实值、预测值 | MSE 与 R2 字典 |
| `save_model/load_model` | `.npz` 路径与参数 | 写入或恢复参数 |
| `save_predictions` | 路径、ID、预测 | `predictions.csv` |
| `save_metrics` | 路径、指标 | `metrics.txt` |
| `run_pipeline` | 六个输入/输出路径 | 完成整条链路 |
| `main` | 命令行参数 | 进程退出码 |

模型保存后必须重新加载再预测，目的是验证磁盘中的模型确实足够恢复推断，而不是只验证当前内存变量。

## 推荐实现与核对顺序

```text
load_regression_csv -> fit_least_squares -> predict -> regression_metrics
-> save_model -> load_model -> save_predictions -> save_metrics -> run_pipeline -> main
```

先从仓库根目录运行：

```powershell
python integrated_tasks/02_linear_regression_csv/demo.py
python integrated_tasks/02_linear_regression_csv/check.py
```

完成全部函数后，再运行：

```powershell
python integrated_tasks/02_linear_regression_csv/starter.py --train integrated_tasks/02_linear_regression_csv/data/train.csv --validation integrated_tasks/02_linear_regression_csv/data/validation.csv --test integrated_tasks/02_linear_regression_csv/data/test.csv --predictions predictions.csv --metrics metrics.txt --model linear_model.npz
```

最终生成 `predictions.csv`、`metrics.txt` 和 `linear_model.npz`。预测保留6位小数；指标键为 `validation_mse`、`validation_r2`。

## 常见错误

- 把 ID 或 target 混进特征；测试集仍按“最后一列是标签”切分；
- 用验证集参与最小二乘拟合；`y` 保留成 `(n,1)`；
- 显式求逆而不是使用 `np.linalg.lstsq`；
- 保存模型后仍用旧内存参数预测；
- 输出表头、键名、顺序、换行或小数位不符合要求；
- 单行 CSV 读成一维 X。

需要完全脱离脚手架时进入 [exam_mode](exam_mode/README.md)。
