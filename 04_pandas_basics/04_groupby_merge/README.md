# pandas 4：分组聚合、表连接与行数审计

## `groupby` 后形状为什么会变

`groupby("course")` 会把相同课程的行放入同一组。聚合后，分组键默认进入索引，不再是普通列。本专题统一使用命名聚合和 `reset_index()`：

```python
summary = (
    scores.groupby("course", dropna=False)
    .agg(
        record_count=("score", "size"),
        student_count=("student_id", "nunique"),
        mean_score=("score", "mean"),
    )
    .reset_index()
)
```

运行前要能说出输入行数、组数、输出列和索引形状。

## `size`、`count` 与 `nunique`

- `size`：组内记录行数，包含目标列缺失的行；
- `count`：某列非缺失值数量；
- `nunique`：不同值数量，适合统计不同学生数。

三者不能只凭名字互换。

## `merge` 的方向

本专题以成绩表为左表、学生信息表为右表：

```text
scores many rows -- student_id --> students one row
```

因此使用 `validate="many_to_one"`。若学生表编号重复，连接会报错，而不是悄悄把每条成绩复制多份。

`_merge` 指示来源：

- `both`：两边都匹配；
- `left_only`：成绩编号在学生表中不存在；
- `right_only`：学生没有成绩，只有 `right` 或 `outer` 连接会保留。

## 多对多为什么会膨胀

某个键左表出现 `a` 次、右表出现 `b` 次，内连接会产生 `a*b` 行。`predict_inner_merge_rows` 在真正连接前按键计算期望行数，让“行数变多”成为可解释结果，而不是意外。

## 完整任务

`build_cohort_course_report` 执行：

```text
读取学生表和成绩表
→ many_to_one 左连接
→ 拒绝未知 student_id
→ 按 cohort/course 分组
→ 记录数、学生数、平均分、及格率
→ 固定排序和严格保存
```

## 学生任务和迁移

1. 手算样例中每个课程的记录数和不同学生数；
2. 去掉 `reset_index()`，比较索引与列；
3. 构造重复学生编号，预测 `validate="many_to_one"` 的结果；
4. 构造左右键各重复两次的表，手算连接后行数；
5. 将连接方式依次改为 inner/left/right/outer，解释 `_merge`；
6. 增加一名无成绩学生，确认严格报告不凭空生成成绩行。
