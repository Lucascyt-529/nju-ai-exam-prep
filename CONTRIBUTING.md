# 仓库维护流程

本仓库虽然主要服务一名学习者，但需要在不同电脑和不同 Codex 对话中保持一致。

## 每批工作的顺序

1. 阅读 `AGENTS.md`、`README.md`、`coverage_matrix.md` 和 `progress.md`。
2. 检查 `git status`，确认没有覆盖用户正在完成的练习。
3. 选择一个足够小、能够在本轮验证的目标。
4. 先写题目、输入输出约定和验收条件，再写参考实现与测试。
5. 运行局部测试；影响公共设施时再运行全量测试。
6. 更新进度、覆盖状态和必要的错题记录。
7. 向用户报告改动、验证结果、假设和下一项练习。

## 专题目录

一个成熟的编程专题通常包含：

```text
topic_name/
├── README.md
├── starter.py
├── reference/
│   └── solution.py
├── tests/
├── data/
└── exercises.md
```

- `starter.py` 由学生完成，不自动覆盖。
- `reference/` 与学生入口隔离，首次做题时不要求阅读。
- 默认测试验证已经完成的公共代码与参考实现。
- 学生练习测试使用专题 README 中给出的独立命令运行。

## 状态更新

`coverage_matrix.md` 只使用以下状态：

- `未开始`
- `已建骨架`
- `已讲解`
- `已实现待验证`
- `已验证`
- `学生已独立完成`

只有代码通过有效测试后才能标记为 `已验证`；只有学生不看答案完成并通过验收后才能标记为 `学生已独立完成`。

## 最低验证

```bash
python -m pytest -q
```

提交前还应检查：

```bash
git status
git diff --check
```

建议保持小提交，例如：

```text
chore: add repository learning controls
feat: add robust csv loading exercises
test: cover standardization edge cases
docs: update chapter 3 progress
```
