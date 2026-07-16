# 练习1：从标准输入读取矩阵

## 题目

第一行包含两个正整数 `n d`，随后有 `n` 行，每行包含 `d` 个浮点数。读取矩阵并输出每一列的平均值，保留6位小数。

输入示例：

```text
3 2
1 2
3 4
5 6
```

期望输出：

```text
3.000000 4.000000
```

## 必须检查

- 第一行是否恰好包含两个整数；
- `n` 和 `d` 是否为正数；
- 实际行数是否等于 `n`；
- 每行列数是否等于 `d`；
- 所有数值是否为有限浮点数；
- 格式错误时是否以非零状态退出，而不是输出一个看似正常的答案。

## 学生任务

完成 `starter.py` 中的四个函数：

1. `parse_matrix()`；
2. `column_means()`；
3. `format_result()`；
4. `main()`。

## 开始编码前

先不要打开 `reference/solution.py`。在 Python 交互环境中运行下面的代码，并在运行前写下三个答案：`X.shape` 是什么、`axis=0` 得到几个数、`axis=1` 得到几个数。

```python
import numpy as np

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

print(X.shape)
print(X.mean(axis=0))
print(X.mean(axis=1))
```

如果无法解释结果，先停在这里学习“二维数组、行、列和 axis”，不要背诵 `mean(axis=0)`。

## 分层完成

1. 先只解析第一行的 `n` 和 `d`；
2. 再把后续行读成 Python 的二维列表；
3. 检查行数和每行列数；
4. 转换为 NumPy 数组并打印 `shape`；
5. 计算列均值并格式化输出；
6. 最后加入错误处理。

先运行学生版本：

```bash
python 00_python_programming/01_standard_input/starter.py < 00_python_programming/01_standard_input/sample_input.txt
```

完成后应与 `expected_output.txt` 一致。参考实现只用于最后核对：

```bash
python 00_python_programming/01_standard_input/reference/solution.py < 00_python_programming/01_standard_input/sample_input.txt
```

## 迁移题

原题通过后，不看参考答案完成以下变化：

1. 输出每一列的最大值；
2. 把输入改为 `2 3`，预测输出形状再运行；
3. 增加一行列数错误的数据，确认程序失败而不是静默计算。

三项都能解释并完成后，才说明你理解了输入形状与 `axis=0`，而不是只记住了原答案。
