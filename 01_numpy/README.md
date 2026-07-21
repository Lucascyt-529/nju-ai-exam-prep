# NumPy

NumPy 是后续机器学习手写实现的计算基础。本模块不要求背诵大量 API，而是要求看到一行数组运算时，能先判断数据含义、输出形状和运算轴。

## 学习顺序

| 顺序 | 专题 | 重点 | 当前学习证据 |
| --- | --- | --- | --- |
| 1 | [数组创建与 dtype](../02_numpy_basics/00_array_creation_dtypes/README.md) | `ndim`、`shape`、`size`、数值类型 | 首轮完成 |
| 2 | [形状、行列与 axis](../02_numpy_basics/01_arrays_shapes_axes/README.md) | 样本轴、特征轴、按轴聚合 | 首轮完成，仍需复验 |
| 3 | [reshape、转置与拼接](../02_numpy_basics/01_arrays_shapes_axes/reshape_practice/README.md) | `(n,)` 与 `(n,1)`、增广矩阵 | 首轮完成，仍需复验 |
| 4 | [索引与筛选](../02_numpy_basics/02_indexing_filtering/README.md) | 列选择、布尔掩码、同步打乱 | 首轮完成 |
| 5 | [广播](../02_numpy_basics/03_broadcasting/README.md) | 按特征/样本运算、复制语义 | 首轮完成 |
| 6 | [矩阵乘法](../02_numpy_basics/04_matrix_multiplication/README.md) | `X @ w`、`X.T @ error`、Gram矩阵 | 首轮完成，随模型复验 |

## 当前使用方式

当前不从头重做全部 NumPy 专题。在线性回归、逻辑回归、PCA 等模型中遇到下列问题时，回到对应专题复习：

- 不确定某个结果是一维还是二维；
- 不确定 `axis=0` 与 `axis=1` 在当前数据中的含义；
- 不确定广播发生在样本轴还是特征轴；
- 不确定矩阵乘法两边维度如何对应；
- 特征和标签经过筛选或打乱后是否仍同步。

## 完成标准

1. 运行前能写出关键中间量的 shape；
2. 能解释每个轴代表样本还是特征；
3. 能在模型代码中正确使用 `X @ w` 和 `X.T @ error`；
4. 能完成一个改变样本数或特征数的变式，而不是只复现原题；
5. 需要严格数据校验时，转入 [文件读写与数据处理](../03_data_io_processing/README.md)，不要把大量校验塞进当前机器学习算法主体。

个人掌握记录以 [LEARNING_STATUS.md](../LEARNING_STATUS.md) 和 [错题本](../records/bug_book/README.md) 为准。
