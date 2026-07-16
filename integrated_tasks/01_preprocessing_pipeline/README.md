# 综合任务1：可恢复的数据预处理流水线

## 场景

没有 OJ 帮你调用函数。你会得到：

- `train.csv`：字符串样本编号、两个可能缺失的浮点特征和整数标签；
- `test.csv`：相同特征，但没有标签。

你需要独立写出完整命令行程序：

```text
读取并检查两个CSV
→ 只从训练集拟合缺失值填充值
→ 填补训练集并拟合均值、标准差
→ 保存参数到.npz
→ 重新加载参数
→ 处理测试集
→ 严格保存结果CSV
→ 自己验证数值、形状和文件格式
```

核心实现只使用 Python 标准库和 NumPy。

## 输入格式

训练集固定表头：

```text
sample_id,feature_1,feature_2,label
```

测试集固定表头：

```text
sample_id,feature_1,feature_2
```

特征空字段表示缺失；样本编号、标签不允许缺失；样本编号在各自文件中必须唯一。

## 输出和参数

结果表头：

```text
sample_id,feature_1,feature_2
```

浮点数保留6位小数。参数文件必须包含三个一维数组：

- `fill_values`：训练集各特征填充值；
- `means`：填补后的训练集均值；
- `scales`：填补后的训练集安全标准差，常数列使用1。

## 运行前手算

查看 `data/train.csv` 和 `data/test.csv`：

1. 手算训练集两个特征的填充值；
2. 写出填补后的训练矩阵；
3. 计算两个特征的均值和总体标准差；
4. 预测测试集三行变换后的结果；
5. 说明为什么测试集中的40和7不能参与拟合参数。

## 参考程序运行

```bash
python integrated_tasks/01_preprocessing_pipeline/reference/solution.py \
  --train integrated_tasks/01_preprocessing_pipeline/data/train.csv \
  --test integrated_tasks/01_preprocessing_pipeline/data/test.csv \
  --output transformed_test.csv \
  --params preprocessing_params.npz
```

在 Windows PowerShell 中可以写成一行。运行后将结果与 `expected/transformed_test.csv` 逐字节比较，并重新加载 `.npz` 检查键名与形状。

## 无 OJ 自测清单

- [ ] 正常样例的手算值一致；
- [ ] 单行测试集仍保持 `(1, 2)`；
- [ ] 测试集极端值不会改变训练参数；
- [ ] 整列缺失训练特征被明确拒绝；
- [ ] 重复编号、错误表头、非法标签被明确拒绝；
- [ ] 输出目录不存在时能创建；
- [ ] 输出表头、列顺序、小数位和换行完全一致；
- [ ] 参数保存后能在新进程中恢复。

## 迁移题

1. 增加第三个特征，让表头和输出列数动态变化；
2. 将均值填补改为中位数填补；
3. 只标准化指定特征，保留另一列原值；
4. 把训练和预测拆成两个独立命令，证明预测命令不读取训练文件；
5. 在此流水线上接入后续线性回归模型。
