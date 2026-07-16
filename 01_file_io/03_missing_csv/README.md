# 文件读写3：使用 genfromtxt 读取缺失值CSV

## 题目

CSV 第一行为表头，第一列是整数样本编号，最后一列是整数标签，中间特征可能有空字段。读取数据，只使用当前训练文件各特征的均值填补缺失值，并保存清洗后的完整CSV。

## 为什么使用 genfromtxt

`np.genfromtxt` 能把空字段转换为 `NaN`，适合规则表格中的缺失数值。需要理解：

- `skip_header=1` 与 `loadtxt` 的 `skiprows=1` 名称不同；
- `missing_values=""` 表示空字段；
- `filling_values=np.nan` 让缺失值先保留，不能悄悄填0；
- 填补统计量必须来自训练数据。

## 运行前预测

查看 `data/train_missing.csv` 后回答：

1. 哪个位置会读成 `NaN`；
2. 第二个特征的训练均值是多少；
3. 清洗后文件应有几行几列；
4. 为什么样本编号和标签不能允许缺失。

## 运行

```bash
python 01_file_io/03_missing_csv/reference/solution.py --input 01_file_io/03_missing_csv/data/train_missing.csv --output cleaned.csv
```

## 迁移题

1. 增加多个缺失位置并手算填补结果；
2. 构造整列缺失，说明为什么均值填补无法继续；
3. 改成中位数填补；
4. 将清洗参数保存下来，再处理另一份测试CSV，禁止重新计算测试均值。
