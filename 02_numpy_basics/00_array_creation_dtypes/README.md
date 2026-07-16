# NumPy 0：数组创建与 dtype

本专题回答最靠前的几个问题：`np.array` 创建的到底是什么、形状从哪里来、为什么一个数组通常只有一种数据类型。

## 一张最小地图

```python
scalar = np.array(3.0)                  # shape == (), ndim == 0
vector = np.array([1.0, 2.0, 3.0])     # shape == (3,), ndim == 1
matrix = np.array([[1.0, 2.0, 3.0]])   # shape == (1, 3), ndim == 2
column = np.array([[1.0], [2.0], [3.0]])  # shape == (3, 1)
```

它们的类型都是 `numpy.ndarray`。区别不在“一层是数组、两层是另一种类型”，而在 `ndim` 和 `shape`。

`dtype` 表示数组元素采用的统一存储类型。机器学习计算通常使用浮点数组，避免整数数组在除法、缺失值和参数更新时产生不符合预期的行为。

## 首次出现的 API

| API | 输入 | 输出 | 本专题用途 |
| --- | --- | --- | --- |
| `np.asarray(values, dtype=float)` | Python 序列 | 浮点数组 | 把列表转成数组 |
| `np.zeros((n, d), dtype=float)` | 目标形状 | 全0二维数组 | 初始化矩阵 |
| `np.arange(start, stop, step)` | 起点、终点、步长 | 一维等差数组 | 创建规则序列 |
| `array.astype(float)` | 已有数组 | 指定类型的新数组 | 明确转换 dtype |

## 运行前预测

先阅读 `guided_demo.py`，不要运行，预测每次输出的 `shape`、`ndim`、`size` 和 `dtype`。特别说明：为什么 `[[1, 2, 3]]` 的 `size` 是3，但 `shape` 是 `(1, 3)`。

然后运行：

```bash
python 02_numpy_basics/00_array_creation_dtypes/guided_demo.py
```

## 学生任务

完成 `starter.py` 的五个小函数。第一轮只要求用清楚直接的 NumPy 写法，不要求把所有检查压缩成一行。

## 迁移题

1. 创建形状为 `(3, 2)` 的全5浮点矩阵；
2. 创建 `[1, 3, 5, 7]`，并说明 `stop` 为什么不被包含；
3. 把整数数组转换为浮点数组，修改新数组后确认原数组是否改变；
4. 分别创建 `(3,)`、`(1, 3)` 和 `(3, 1)`，用自己的话解释三个形状。
