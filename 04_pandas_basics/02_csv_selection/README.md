# pandas 2：读取 CSV 与选择数据

## 完整的小流程

```text
读取 CSV → 检查 shape/columns/dtypes → 选择需要的列 → 筛选行 → 转入 NumPy
```

本专题的数据文件是 `data/students.csv`。它同时包含字符串和数值，适合体会 pandas 相比纯 NumPy 读取表格的便利。

## 首次出现的 API

- `pd.read_csv(path)`：读取带表头 CSV；
- `frame.iloc[start:stop]`：按整数位置选择行；
- `frame.loc[mask, columns]`：按标签和布尔条件选择；
- `frame.dtypes`：检查每列类型。

`.iloc` 看的是第几个位置，`.loc` 看的是索引标签或列名。默认索引恰好也是整数时，两者显示可能相似，但含义不同。

## 运行前预测

阅读 `guided_demo.py` 后回答：

1. 数据有多少行、多少列；
2. `frame.iloc[1:3]` 保留哪两名学生；
3. `frame.loc[frame["score"] >= 85, ["name", "score"]]` 的形状；
4. 筛选后的原始索引是否自动变成从0开始。

## 学生任务和迁移

完成 `starter.py` 后，把筛选条件改为“学习时长至少3小时且出勤率至少0.9”。再将选出的两个特征转换成二维浮点 NumPy 数组，并检查形状。
