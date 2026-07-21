# Learner-first 第一阶段迁移记录

## 范围

- 分支：`refactor/learner-first-v1`
- 日期：2026-07-21
- 只迁移普通线性回归并建立学习者优先骨架，不迁移其他算法。

## 移动与重命名

- 原根目录 `README.md` -> `docs/project_management/original_build_spec.md`
- `PROJECT_STATE.md` -> `docs/project_management/BUILD_STATUS.md`
- `progress.md` -> `docs/project_management/legacy_progress.md`
- `coverage_matrix.md` -> `docs/curriculum/full_coverage_matrix.md`
- `foundation_coverage.md` -> `docs/curriculum/foundation_coverage.md`
- `learning_method.md` -> `docs/curriculum/learning_method.md`
- `syllabus.md` -> `docs/curriculum/syllabus.md`
- `exam_evidence.md` -> `docs/exam/exam_evidence.md`
- `historical_baseline.md` -> `docs/project_management/historical_baseline.md`
- `bug_book/` -> `records/bug_book/`
- 普通线性回归参考实现 -> `02_machine_learning/01_linear_regression/reference/solution.py`

## 重写与新增

- 重写根目录 `README.md`，只保留一个当前学习入口。
- 新建 `LEARNING_STATUS.md`，把个人掌握与仓库建设分开。
- 更新 `AGENTS.md` 的恢复路径、简单事实解释规则及学习/模拟模式边界。
- 将 `START_HERE.md` 改为旧链接兼容说明。
- 建立 `00_getting_started/` 至 `06_mock_exams/`、`resources/`、`records/` 和 `docs/` 导航。
- 线性回归新增独立 README、练习骨架、`demo.py`、`check.py` 和三层练习说明。
- 新增新入口运行测试，严格参考测试改为加载新路径。
- 后续按学习者反馈进一步简化：demo 只给输入、期望输出和实际输出；check 按函数顺序做数值核对；starter 移除基础 shape/内容校验，机器学习阶段先练算法主线。

## 删除与兼容策略

- 删除旧普通线性回归 `guided_demo.py`，其功能由新 `demo.py` 覆盖。
- 旧普通线性回归目录采用方案 A：只保留跳转 README，不保留第二份普通线性回归实现。
- 尚未迁移的 `generalized_linear_models/` 暂留旧目录，并在跳转 README 中明确标注。
- 重构前已填写的学生 starter 备份为 `records/migration_backups/linear_regression_starter_before_refactor.py`。

## 执行前基线

```text
python -m pytest -q
```

结果：收集阶段 1 项错误。`tests/test_linear_regression_reference.py` 仍指向已经移动的旧参考实现路径；这是接手时半迁移状态的已知断点。

## 最终验证

```text
python 02_machine_learning/01_linear_regression/demo.py
```

结果：退出码0；打印一组 `X、w、b`、期望预测和实际预测。当前 starter 未填写，因此清晰提示 `predict` 尚未完成，无 traceback。

```text
python 02_machine_learning/01_linear_regression/check.py
```

结果：按实现顺序打印期望值和实际值，遇到第一个未完成函数停止。当前按约定退出码1；自动测试将参考实现注入后5/5通过。

```text
python -m pytest -q tests/test_repository_structure.py tests/test_linear_regression_reference.py tests/test_linear_regression_learning_entry.py
```

结果：`20 passed in 1.51s`。

```text
python -m pytest -q
```

结果：`2106 passed in 34.70s`。

另执行 Markdown 相对链接审计，结果为 `markdown_links=OK`。

## 暂缓内容

- `watermelon_book/` 中进阶变式的批量搬迁；
- 两套现有模拟卷重写；
- 全量严格测试改造成学习者检查；
- 旧基础目录和综合任务的物理迁移。

## Learner-first 第二阶段

日期：2026-07-21

- 将模型评估、逻辑回归、kNN、PCA、K-means、离散决策树、朴素贝叶斯、感知机、LDA 和线性 SMO 的学生骨架与参考实现迁入 `02_machine_learning/`。
- 将 AdaBoost、Bagging/OOB 和随机森林归入统一的集成学习入口。
- 为上述专题新增只展示输入、期望输出和实际输出的 `demo.py`，以及按主线数值核对的 `check.py`。
- 旧核心目录改为跳转 README；严格演示迁为 `reference_demo.py`，原章节中的剪枝、牛顿法、多分类、核方法等扩展保留原位。
- 更新全部相关严格测试和章节链接，使实现只有一个真实来源。
- 新增统一入口测试，验证日常文件齐全、demo 可运行、check 样例与参考实现一致。

验证结果：机器学习入口与迁移算法专项423项通过；全量2145项通过。个人学习状态仍停在线性回归，本次建设不计为学习者掌握。

## Learner-first 第三阶段

日期：2026-07-21

- 学习者指出除线性回归外的算法 README 明显过短，只有流程导航，不能支持独立自学；审计确认这些入口原先仅18～33行，而线性回归为138行。
- 将模型评估、逻辑回归、kNN、PCA、K-means、决策树、朴素贝叶斯、感知机、AdaBoost、Bagging/OOB、随机森林、LDA 和 SVM 扩展为101～158行的专题讲义。
- 每篇按算法特点补齐输入输出、公式、函数职责、实现顺序、手算样例、评估方法、常见错误和自学闭环；没有改写学习者 starter。
- 集成学习总览补充三条路线的训练依赖、随机性来源和评价方式，明确 Boosting、Bagging 与随机森林的区别。
- 新增 README 教学结构测试，防止入口退化为只有流程图和运行命令的短说明。

验证结果：入口专项55项通过；全量2158项通过；108个 Markdown 相对链接全部有效。个人学习状态仍停在线性回归。

## Learner-first 第四阶段

日期：2026-07-21

- 学习者希望 README 在实现细节前提供接近周志华《机器学习》知识组织方式的算法介绍，以便首次接触时先形成基本认识。
- 为线性回归、逻辑回归、kNN、PCA、K-means、决策树、朴素贝叶斯、感知机、AdaBoost、Bagging/OOB、随机森林、LDA 和 SVM 新增简短“算法概览”。
- 概览分别说明算法解决的问题、核心假设或动机、主要能力与局限，以及与前后算法的关系；全部为转述，不复制教材原文。
- 模型评估使用“基本认识”说明经验误差、泛化和模型选择；集成学习总览说明准确性与多样性的共同作用。
- 新增14项结构检查，保证具体算法概览和两个总览不会在后续重写中丢失。

验证结果：入口专项69项通过；全量2172项通过；Markdown 相对链接全部有效。未修改学习者 starter，未改变个人学习状态。
