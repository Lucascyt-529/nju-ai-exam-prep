# 项目续接状态

本文件是新会话恢复工作的第一入口。个人掌握只看根目录 `LEARNING_STATUS.md`，教材覆盖只看 `docs/curriculum/full_coverage_matrix.md`。

## 项目目标

为南京大学人工智能学院预推免机试建设可独立作答的训练仓库，覆盖完整 Python 程序、文件读写、NumPy 数据处理、无 OJ 自检，以及周志华《机器学习》第1～10章主线的 NumPy 手写实现。第11～16章在前十章完成前暂缓。

## 当前状态

- 日期：2026-07-21
- 当前分支：`main`
- 远程：`Lucascyt-529/nju-ai-exam-prep`
- 工作批次：learner-first 有限修补 A～F
- 当前学习入口：`02_machine_learning/01_linear_regression/README.md`
- 个人学习位置：线性回归；新建材料不代表学习者已经掌握

恢复时必须重新运行 `git status` 和 `git log -5 --oneline`，不要把本文件当作提交状态证明。

## 当前结构决定

- 六大顶层学习导航保持不变，不再机械搬迁旧专题。
- 学生 `starter.py`、参考实现、demo/check 和严格测试分离。
- 当前机器学习阶段先练预测、损失/指标、训练和验证；学生算法代码假定输入合法。
- 原物理路径继续承载稳定实现，新增入口优先复用既有算法逻辑。
- 根 README 保持线性回归为唯一当前专题。
- exam mode 只暴露题面、数据、输出、时间和评分，不提供函数签名或解题步骤。

## 最近完成

- 线性回归补齐 shape 约定、完整中间量 demo、12项日常 check 和分层练习。
- 模型评估新增数据划分与二分类指标学习入口，复用已验证的留出和指标实现。
- 算法题新增滑窗、前缀和、单调栈、Top K、链表、二叉树、回溯、区间贪心、拓扑排序九种核心模式；每个专题只保留一个代表任务。
- 线性回归、逻辑回归、kNN、PCA、K-means 增加无脚手架 exam mode。
- 线性回归 CSV 综合题新增学生可见 demo/check、完整调用图和独立 exam mode；现有学生代码未改写。
- 旧的千行建设日志原样归档到 `history/BUILD_STATUS_through_2026-07-21.md`。

## 当前测试基线

- A 阶段线性回归：16 passed。
- B 阶段模型评估：26 passed。
- C 阶段算法核心模式：11 passed。
- D 阶段 exam mode：1 passed。
- E 阶段 CSV 综合题：10 passed。
- 最终导航、结构与学习入口验收：90 passed。
- 全量测试：2186 passed in 35.82s。
- 全仓 Python 编译、Markdown 相对链接与 `git diff --check`：通过。

## 下一步

1. 学习者继续完成线性回归 `fit_least_squares`，运行日常 check。
2. 从空文件完成线性回归 exam mode，并做双特征或带噪声变式。
3. 在线性回归 CSV 综合题继续完成模型保存、恢复、输出与 `main()`。
4. 通过线性回归迁移验收后进入数据划分、分类指标与逻辑回归。

## 阻塞项

- 当前无仓库建设阻塞。
- 教材第11～16章按长期优先级主动暂缓，不属于阻塞。

## 历史索引

- [截至 2026-07-21 的完整建设日志](history/BUILD_STATUS_through_2026-07-21.md)
- [迁移日志](migration_log.md)
- [Learner-first 验收](learner_first_acceptance.md)
