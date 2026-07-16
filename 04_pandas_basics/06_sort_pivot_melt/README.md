# pandas 6：稳定排序、透视与长宽表转换

## 稳定排序

当排序键相同时，稳定排序保留这些行原来的相对顺序：

```python
frame.sort_values(by=["cohort", "score"], ascending=[True, False], kind="stable")
```

排序会重新排列行，但默认保留原索引。本专题用参数明确决定是否 `reset_index(drop=True)`，不能把“行位置”和“索引标签”混为一谈。

## `pivot`：每个格子只能有一个值

长表中 `(student_id, course)` 必须唯一，才能严格变成宽表：

```text
student_id, course, score
S001,       math,   80
S001,       python, 90
```

若同一学生同一课程有两行，`pivot` 无法判断选哪一个。本专题先显式检查重复键并报错。

## `pivot_table`：允许重复，但必须说明怎样聚合

`pivot_table(..., aggfunc="mean")` 会把重复格子求均值。它不是“更宽容的 pivot”，而是语义不同的聚合操作。参考实现只允许显式选择 `mean`、`sum`、`max` 或 `min`。

## `melt`：宽表回到长表

`melt` 把课程列恢复为 `course` 和 `score` 两列。缺失格子是否保留由 `drop_missing` 明确控制。转换后统一稳定排序和重置索引。

## 容易忽略的元数据

`pivot` 后列索引可能保留 `columns.name="course"`。参考实现将其清空，并返回普通 `RangeIndex`，避免保存和列比较时出现隐藏差异。

## 学生任务和迁移

1. 构造分数相同的三行，验证稳定排序保留原顺序；
2. 比较排序后保留原索引与重置索引；
3. 手算长表转宽表后的形状和列顺序；
4. 增加重复 `(student_id, course)`，分别运行 pivot 和 mean pivot_table；
5. 删除一个学生课程组合，观察宽表缺失格子；
6. 将完整宽表 melt 回长表，与规范排序后的原表逐行比较。
