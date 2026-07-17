# 基础能力覆盖审计

本文件跟踪教材之外、但完成机试完整程序必需的基础能力。它记录仓库建设证据，不等于学习者掌握；个人掌握只以`progress.md`中的独立作答证据为准。

## Python与文件读写

| 能力 | 当前证据 | 建设状态 | 学习状态 |
| --- | --- | --- | --- |
| 标准输入、循环读行、维度检查 | `00_python_programming/01_standard_input/` | 已验证 | 尚未独立完成 |
| UTF-8文本数值表 | `01_file_io/01_text_table/` | 已验证 | 尚未独立完成 |
| 纯数值CSV与`loadtxt` | `01_file_io/02_clean_csv/` | 已验证 | 尚未独立完成 |
| 缺失CSV与`genfromtxt` | `01_file_io/03_missing_csv/` | 已验证 | 尚未独立完成 |
| 混合类型CSV与逐行校验 | `01_file_io/04_mixed_csv/` | 已验证 | 尚未独立完成 |
| `.npy/.npz`、精确路径、严格模型恢复 | `01_file_io/05_numpy_model_files/` | 已验证 | 尚未独立完成 |
| 命令行参数、错误码、输出目录 | `00_python_programming/02_command_line_program/`及综合任务 | 已验证 | 尚未独立完成 |

## NumPy

| 能力 | 当前证据 | 建设状态 | 学习状态 |
| --- | --- | --- | --- |
| 创建、dtype、`ndim/shape/size` | `02_numpy_basics/00_array_creation_dtypes/` | 已验证 | 尚未独立完成 |
| 行列、axis与索引降维 | `01_arrays_shapes_axes/` | 已验证 | 完成首次口头诊断，未稳定掌握 |
| reshape、转置与拼接 | `01_arrays_shapes_axes/reshape_practice/` | 已验证 | 尚未独立完成 |
| 索引、布尔筛选、同步打乱 | `02_indexing_filtering/` | 已验证 | 尚未独立完成 |
| 广播与安全标准化 | `03_broadcasting/` | 已验证 | 完成首次广播解释，未稳定掌握 |
| 矩阵乘法、线性输出与Gram矩阵 | `04_matrix_multiplication/` | 已验证 | 尚未独立完成 |

## 数据处理与pandas

| 能力 | 当前证据 | 建设状态 | 学习状态 |
| --- | --- | --- | --- |
| 训练集均值填补与标准化 | `03_data_processing/01_imputation_standardization/` | 已验证 | 尚未独立完成 |
| 训练词表、未知类别、one-hot与JSON元数据 | `02_categorical_encoding/` | 已验证 | 尚未独立完成 |
| 留出、分层、K折、Bootstrap | `watermelon_book/02_model_evaluation/01_data_splitting/` | 已验证 | 尚未独立完成 |
| Series/DataFrame、CSV、清洗 | `04_pandas_basics/01_...03_...` | 已验证 | 尚未独立完成 |
| groupby/merge、NumPy桥接、pivot/melt | `04_pandas_basics/04_...06_...` | 已验证 | 尚未独立完成 |
| 完整预处理流水线 | `integrated_tasks/01_preprocessing_pipeline/` | 已验证 | 前置基础未完成，暂不作答 |

## 完整程序与无OJ验证

| 类型 | 当前证据 | 建设状态 |
| --- | --- | --- |
| 预处理完整程序 | `integrated_tasks/01_preprocessing_pipeline/` | 已验证 |
| 线性回归CSV程序 | `integrated_tasks/02_linear_regression_csv/` | 已验证 |
| 逻辑回归CSV程序 | `integrated_tasks/03_logistic_regression_csv/` | 已验证 |
| K-means无监督CSV程序 | `integrated_tasks/04_kmeans_csv/` | 已验证 |
| 120分钟混合诊断卷 | `mock_exams/01_foundation_diagnostic/` | 参考与评分已验证，学习者未作答 |

## 仍需建设的基础缺口

按当前优先级：

1. mini-batch索引、可复现随机数与训练循环数据流；
2. 异常值诊断和仅用训练集拟合的稳健处理边界；
3. 类别词表JSON与NumPy模型包组合成一份完整混合特征预测任务；
4. 学习者完成基础starter后，建立真正的个人掌握证据和错题记录。

后续每补一项都必须有讲解、学生入口、参考实现、边界测试和至少一道迁移题。
