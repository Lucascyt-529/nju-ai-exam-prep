# 文件读写与数据处理

本模块把“题目数据变成可训练数组，再把结果按要求保存”作为一条完整能力训练。机器学习核心算法可以先假定数组合法；到了这里，再集中处理文件结构、缺失值、类型、训练参数复用和严格输出。

## 第一层：完整 Python 程序

| 顺序 | 专题 | 训练目标 |
| --- | --- | --- |
| 1 | [标准输入程序](../00_python_programming/01_standard_input/README.md) | `main()`、逐行读取、标准输出与错误退出 |
| 2 | [命令行程序](../00_python_programming/02_command_line_program/README.md) | 参数、路径、输出目录和可测试流水线 |

## 第二层：文件读写

| 顺序 | 专题 | 训练目标 |
| --- | --- | --- |
| 1 | [UTF-8 文本数值表](../01_file_io/01_text_table/README.md) | 空行、列数、数值转换和结果文件 |
| 2 | [纯数值 CSV](../01_file_io/02_clean_csv/README.md) | 表头、`loadtxt`、特征/标签切分、`savetxt` |
| 3 | [缺失值 CSV](../01_file_io/03_missing_csv/README.md) | `genfromtxt`、缺失值和严格格式 |
| 4 | [混合类型 CSV](../01_file_io/04_mixed_csv/README.md) | 标准库 `csv`、字符串字段与逐行解析 |
| 5 | [NumPy 模型文件](../01_file_io/05_numpy_model_files/README.md) | `.npy/.npz`、模型键和安全恢复 |

## 第三层：NumPy 数据处理

所有 `fit` 参数只从训练集估计，验证集和测试集只执行 `transform`。

| 顺序 | 专题 | 训练目标 |
| --- | --- | --- |
| 1 | [填补与标准化](../03_data_processing/01_imputation_standardization/README.md) | 训练均值、尺度、常数列与数据泄漏 |
| 2 | [类别编码](../03_data_processing/02_categorical_encoding/README.md) | 训练词表、未知桶、one-hot 与拼接 |
| 3 | [Mini-batch](../03_data_processing/03_minibatch_training/README.md) | 批下标、同步取样与随机种子 |
| 4 | [异常值处理](../03_data_processing/04_outlier_handling/README.md) | IQR、裁剪、稳健缩放与训练参数复用 |

## 第四层：pandas 表格处理

pandas 用于有表头、混合类型和缺失值的表格；模型核心计算仍回到 NumPy。

| 顺序 | 专题 | 训练目标 |
| --- | --- | --- |
| 1 | [Series 与 DataFrame](../04_pandas_basics/01_series_dataframe/README.md) | 索引、列和 dtype |
| 2 | [CSV 与选择](../04_pandas_basics/02_csv_selection/README.md) | `.loc`、`.iloc` 和布尔筛选 |
| 3 | [清洗](../04_pandas_basics/03_cleaning/README.md) | 空值、非法数字、重复键与类型转换 |
| 4 | [groupby 与 merge](../04_pandas_basics/04_groupby_merge/README.md) | 聚合、连接和行数审计 |
| 5 | [pandas/NumPy 桥接](../04_pandas_basics/05_pandas_numpy_bridge/README.md) | 列顺序、dtype、shape 和训练统计量 |
| 6 | [排序与长宽表](../04_pandas_basics/06_sort_pivot_melt/README.md) | 稳定排序、pivot 和 melt |

## 推荐学习方式

1. 先根据当前综合题暴露的问题选择一个基础专题；
2. 读 README 并预测输入输出；
3. 独立完成 `starter.py`；
4. 用样例和严格测试核对文件内容；
5. 回到综合题，把读取、处理、模型和保存串成完整程序。

当前对应的第一个综合闭环是 [线性回归 CSV](../integrated_tasks/02_linear_regression_csv/README.md)。
