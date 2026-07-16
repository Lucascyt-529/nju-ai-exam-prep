# pandas 1：Series 与 DataFrame

## 两个核心对象

```python
scores = pd.Series([78.0, 91.0], index=["小林", "小周"], name="score")
```

`Series` 是一维数据，同时带有索引。`scores.shape == (2,)`。

```python
students = pd.DataFrame({
    "name": ["小林", "小周"],
    "study_hours": [2.5, 4.0],
    "score": [78.0, 91.0],
})
```

`DataFrame` 是二维表格，行索引和列名都是对象的一部分。`students.shape == (2, 3)`。

## 运行前预测

阅读 `guided_demo.py`，先回答：

1. `scores["小周"]` 返回一个数还是一个 Series；
2. `students["score"]` 与 `students[["score"]]` 的形状分别是什么；
3. 为什么 `name` 列与两个数值列的 dtype 不同；
4. 转成 NumPy 时为什么要先选择数值列。

## 学生任务

完成 `starter.py`。每个函数完成后分别打印 `.shape`、`.index`、`.columns` 或 `.dtype`，不要只看表格显示效果。

## 迁移题

1. 增加一列出勤率并保持浮点类型；
2. 把列顺序改为 `name, score, study_hours`；
3. 对比选择单列时一层和两层中括号的返回类型与形状；
4. 修改转换后的 NumPy 数组，检查原 Series 是否改变。
