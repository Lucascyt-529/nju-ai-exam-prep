# 综合任务5：混合特征分类与独立模型恢复

## 任务目标

把此前分开的基础能力串成一条无OJ完整链路：

```text
混合CSV
→ 数值空白解析为NaN，类别字符串保持原义
→ 只用训练集拟合均值填补、均值/标准差和类别词表
→ 数值标准化 + 类别one-hot未知桶
→ NumPy对数几率回归
→ 验证集多指标
→ encoder.json + model.npz
→ 新进程只读取状态和test.csv完成预测
→ 严格predictions.csv
```

核心程序只使用Python标准库和NumPy，不使用pandas或sklearn。

## 数据格式

训练和验证表头必须按顺序为：

```text
sample_id,age,monthly_spend,city,device,label
```

测试集没有 `label`。`age`、`monthly_spend` 允许空白，表示缺失；显式
`NaN/Inf` 不允许。`city`、`device` 不允许空白。ID必须非空且唯一，标签只能
是字符 `0` 或 `1`。

## 为什么需要两个状态文件

- `encoder.json` 保存特征名、顺序、每个类别列的训练词表和未知桶策略；
- `model.npz` 保存数值填充值、均值、尺度、模型权重、截距和阈值。

JSON适合人工检查类别语义；NPZ适合精确保存数值数组。预测时两者必须兼容：
如果词表决定编码后有8列，而权重只有7个，程序必须拒绝，不能靠广播或截断继续。

类别词表和数值统计量都只能来自训练集。验证/测试的新类别进入每个特征块最后的
`<UNKNOWN>` 列，不得偷偷扩充词表，否则模型权重列的含义会改变。

## 两个独立命令

训练：

```bash
python integrated_tasks/05_mixed_feature_classification/reference/solution.py train \
  --train integrated_tasks/05_mixed_feature_classification/data/train.csv \
  --validation integrated_tasks/05_mixed_feature_classification/data/validation.csv \
  --encoder work/encoder.json \
  --model work/model.npz \
  --metrics work/metrics.txt
```

预测：

```bash
python integrated_tasks/05_mixed_feature_classification/reference/solution.py predict \
  --test integrated_tasks/05_mixed_feature_classification/data/test.csv \
  --encoder work/encoder.json \
  --model work/model.npz \
  --predictions work/predictions.csv
```

在PowerShell中可把反斜杠续行改成一整行执行。

## 学生分关顺序

1. 严格读取CSV并分别打印数值、类别、标签的形状；
2. 只在训练集拟合数值预处理状态；
3. 只在训练集拟合逐列类别词表；
4. 手算一个已知类别和一个未知类别的one-hot位置；
5. 拼接设计矩阵并确认每一列语义；
6. 完成对数几率回归训练与验证指标；
7. 保存并严格恢复JSON/NPZ；
8. 关闭训练流程，只运行独立预测命令；
9. 逐字节比较输出。

不要直接复制参考程序。每关先写形状、手算两行，再实现对应的starter函数。

## 无OJ验收

- [ ] 缺失数值使用训练填充值；
- [ ] 测试集均值与类别不进入任何拟合状态；
- [ ] 每个类别特征都有独立未知桶；
- [ ] 编码列数与权重长度完全一致；
- [ ] JSON键、版本、特征顺序和词表严格校验；
- [ ] NPZ使用 `allow_pickle=False` 且严格校验键和形状；
- [ ] 单行测试保持二维设计矩阵；
- [ ] 训练和预测可以在两个进程中完成；
- [ ] 概率为有限 `[0,1]` 数值，输出固定6位小数和LF换行。

## 迁移题

1. 新增一个类别特征，列出JSON、one-hot宽度和权重形状需要改动的位置；
2. 把测试集所有city改成新类别，解释为什么词表文件不变；
3. 把训练集一个数值列全改为空白，说明应在哪一步失败；
4. 故意交换JSON中的类别特征顺序，验证恢复阶段拒绝；
5. 将训练与预测放在两个不同目录运行，证明路径不依赖当前工作目录；
6. 保存one-hot列名清单，检查模型中最大绝对权重对应的真实特征语义。
