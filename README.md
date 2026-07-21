# 南京大学人工智能学院机试训练

本仓库用于练习三类完整机试任务：

1. 文件读取、数据处理与结果输出；
2. 完整算法题；
3. 机器学习训练、预测与模型评估。

## 当前学习

**当前学习：线性回归**<br>
**唯一入口：[02_machine_learning/01_linear_regression/README.md](02_machine_learning/01_linear_regression/README.md)**

个人掌握情况见 [LEARNING_STATUS.md](LEARNING_STATUS.md)。仓库建设、教材覆盖和历史记录统一放在 [docs/](docs/) 中，不作为日常学习入口。

## 总学习路线

```text
NumPy 按需复习
-> 机器学习算法与模型评估
-> 文件读写与数据处理
-> 算法迁移
-> 综合题
-> 全真模拟
```

模型评估位于机器学习模块中，并随每个模型一起练习。

## 六大入口

- [NumPy](01_numpy/README.md)
- [机器学习](02_machine_learning/README.md)
- [文件读写与数据处理](03_data_io_processing/README.md)
- [算法题](04_algorithms/README.md)
- [综合练习](05_integrated_practice/README.md)
- [模拟机试](06_mock_exams/README.md)

第一次在本机运行仓库时，先看 [开始使用](00_getting_started/README.md)。

## 建设状态与学习状态

| 模块 | 仓库材料 | 个人学习状态 |
| --- | --- | --- |
| NumPy | 6个基础专题均有讲解、学生入口和参考验证 | 已完成首轮，随模型复验 |
| 机器学习 | 13个主线入口与进阶扩展已建立 | 正在线性回归 |
| 文件与数据处理 | Python程序、文件、预处理、pandas 共17个专题 | 按综合任务暴露问题后复验 |
| 算法题 | 9个完整程序专题 | 尚未进入本轮训练 |
| 综合练习 | 5个端到端任务 | 正在进入线性回归 CSV 任务 |
| 模拟机试 | 2套120分钟、150分模拟卷 | 尚未计时作答 |

“仓库材料已建成”不代表“学习者已经掌握”。当前位置只以本页的唯一入口和 [LEARNING_STATUS.md](LEARNING_STATUS.md) 为准。

## 今天如何学习

1. 阅读当前专题的 `README.md`；
2. 运行 `demo.py`，核对输入和输出；
3. 按推荐顺序完成 `starter.py`；
4. 运行 `check.py`；
5. 完成至少一道变式；
6. 确实需要时再查看 `reference/solution.py`。

```powershell
python 02_machine_learning/01_linear_regression/demo.py
python 02_machine_learning/01_linear_regression/check.py
```

## 管理与旧内容

- [docs/](docs/) 保存建设状态、覆盖矩阵和历史材料，不作为日常入口；
- [records/](records/) 保存经确认的错题与学习记录；
- 旧专题物理路径在不产生重复实现的前提下保留，六大入口是稳定的学习导航；
- 已迁移的机器学习核心旧入口已经删除，进阶内容继续由 [advanced](02_machine_learning/advanced/README.md) 导航。
