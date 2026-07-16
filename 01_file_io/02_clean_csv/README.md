# 文件读写2：使用 loadtxt 读取完整CSV

## 题目

CSV 第一行为表头，后续全部是完整数值。第一列为样本编号，最后一列为标签，中间列为特征。读取文件，计算每个样本所有特征之和，输出：

```text
sample_id,feature_sum
```

## 为什么使用 loadtxt

`np.loadtxt` 适合结构规则、没有缺失值的纯数值数据。需要明确指定：

- `delimiter=","`；
- `skiprows=1` 跳过表头；
- `ndmin=2` 保证单行数据仍是二维。

## 运行前预测

查看 `data/train.csv` 后回答：

1. 完整数组形状是什么；
2. 样本编号、特征矩阵和标签的形状分别是什么；
3. 为什么 `X.sum(axis=1)` 是每个样本一个结果；
4. 输出为什么要分别使用整数和6位浮点格式。

## 运行

```bash
python 01_file_io/02_clean_csv/reference/solution.py --input 01_file_io/02_clean_csv/data/train.csv --output result.csv
```

## 迁移题

1. 将标签列移动到第二列并修改切分逻辑；
2. 输入只保留一个样本，确认 `X` 仍为二维；
3. 将特征和改为特征均值；
4. 构造包含空字段的文件，观察 `loadtxt` 为什么不适合，再转到下一专题。
