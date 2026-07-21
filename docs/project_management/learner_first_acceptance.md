# Learner-first 重构验收

## 验收结论

Learner-first 目录重构已经完成。根目录只承担学习导航，个人学习、仓库建设、教材覆盖、参考实现和历史记录已经分开。尚未学习的内容不因仓库存在参考实现而标记为掌握。

## 逐项证据

| 要求 | 当前证据 | 结论 |
| --- | --- | --- |
| 根目录有唯一当前专题 | `README.md` 只指定线性回归，`LEARNING_STATUS.md` 单独记录个人进度 | 通过 |
| 日常入口数量受控 | 根目录只列 NumPy、机器学习、数据处理、算法题、综合练习、模拟机试六大入口 | 通过 |
| 管理文档退出首页 | 建设、覆盖、考试证据和历史统一位于 `docs/` | 通过 |
| 学生与参考代码分离 | 日常专题使用 `starter.py`，严格答案位于 `reference/solution.py` | 通过 |
| 机器学习适合自学 | 主线算法有概览、公式、函数职责、手算例、常见错误、自学闭环、demo 和 check | 通过 |
| 简单核对不过度工程化 | 机器学习 demo/check 只核对主线数值；shape 和内容校验后移到数据处理 | 通过 |
| 其余入口不是占位页 | NumPy、数据处理、算法题、综合练习和模拟机试均列出实际专题、顺序与启用条件 | 通过 |
| 模拟模式不泄露解题框架 | 新入口只指向开始说明、试卷和数据；计时前禁止查看参考、期望和评分器 | 通过 |
| 旧内容选择性退役 | 旧机器学习核心只留跳转；其余旧总目录页改为兼容入口，稳定专题路径继续承载实现与测试 | 通过 |
| 现有学习者代码不被覆盖 | 迁移前线性回归代码有备份；当前 CSV 学生文件保持本地未提交修改 | 通过 |
| 链接与结构可自动验证 | `test_learner_first_navigation.py`、入口测试和全仓 Markdown 链接测试 | 通过 |

## 保留旧物理路径的原因

`00_python_programming/`、`01_file_io/`、`02_numpy_basics/`、`03_data_processing/`、`04_pandas_basics/`、`algorithms/`、`integrated_tasks/` 和 `mock_exams/` 已经拥有大量经过验证的数据、参考实现和测试引用。把它们机械搬到六大入口下面不会改善学习体验，却会制造大范围路径变更，并可能影响正在填写的学生文件。

因此采用“新目录负责稳定学习导航，原专题路径负责内容存储”的方式。旧总目录 README 已退役为兼容跳转，不再形成第二套学习路线。这不是未完成迁移，而是明确的最终目录决策。

## 不属于本次重构的长期工作

- 教材第11～16章仍按既定优先级暂缓；
- 学习者尚未完成的专题仍按实际作答逐步推进；
- 新模拟卷和综合题只在真实学习暴露新需求后增加；
- 这些属于课程建设或学习进度，不阻塞 learner-first 重构验收。

## 发布前验收命令

```powershell
python -m pytest -q tests/test_learner_first_navigation.py tests/test_repository_structure.py tests/test_machine_learning_learning_entries.py tests/test_linear_regression_learning_entry.py
python -m pytest -q
git diff --check
```

实际运行结果记录在 [BUILD_STATUS.md](BUILD_STATUS.md) 的“最近验证基线”。
