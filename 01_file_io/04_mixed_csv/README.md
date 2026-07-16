# 文件读写4：标准库 csv 处理混合类型表格

## 为什么不能直接用 loadtxt

下面的表格同时包含字符串和数值：

```text
sample_id,name,feature_1,feature_2,label
S001,小林,1.0,10.0,0
```

`np.loadtxt(..., dtype=float)` 无法把 `S001` 和姓名转换成浮点数。这里使用标准库 `csv.DictReader` 逐行读取，再只把指定数值列转换成 NumPy 数组。

## 题目

读取固定表头的混合类型 CSV，返回：

- `sample_ids`：字符串列表；
- `names`：字符串列表；
- `X`：形状为 `(n_samples, 2)` 的浮点数组；
- `y`：形状为 `(n_samples,)` 的整数数组。

计算每个样本两个特征的均值，严格输出：

```text
sample_id,name,feature_mean,label
```

## 必须处理

- UTF-8 表头及固定列顺序；
- 空文件、缺字段和多余字段；
- 空编号、空姓名和重复编号；
- 非数值、NaN 和无穷大；
- 非整数标签；
- 输出目录、列顺序、6位小数和 `\n` 换行。

## 运行前预测

查看 `data/students.csv` 后先写出 `X`、`y` 和每行均值的值与形状，再运行：

```bash
python 01_file_io/04_mixed_csv/reference/solution.py --input 01_file_io/04_mixed_csv/data/students.csv --output mixed_summary.csv
```

## 迁移题

1. 将姓名改成包含逗号的 `"林,晓"`，说明为什么 `csv` 模块比手写 `line.split(",")` 安全；
2. 新增第三个特征，并让代码根据列名选择特征；
3. 把标签移到第二列，修改表头检查和提取逻辑；
4. 构造重复编号和缺字段文件，确认错误信息能定位行号。
