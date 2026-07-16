# NumPy 4：矩阵乘法与线性输出

## 先做形状推理

如果：

```text
X.shape == (n, d)
w.shape == (d,)
```

那么 `X @ w` 的形状是 `(n,)`。内侧的 `d` 对齐并被消去，剩下每个样本一个线性输出。

如果 `W.shape == (d, k)`，那么 `X @ W` 的形状是 `(n, k)`，表示每个样本有 `k` 个输出。

## 运行前预测

阅读 `guided_demo.py`，先回答：

1. `X @ w` 的两个值分别是多少；
2. `X.T.shape` 是什么；
3. `X.T @ X` 的形状为什么是 `(d, d)`；
4. `X * w` 与 `X @ w` 的结果形状和含义有何不同。

然后运行：

```bash
python 02_numpy_basics/04_matrix_multiplication/guided_demo.py
```

## 学生任务

完成：

- `matrix_product()`；
- `linear_scores()`；
- `multi_output_scores()`；
- `feature_gram_matrix()`。

每次写 `@` 之前，先在注释中写出左右两边形状和预期输出形状。

## 迁移题

1. 给 `X` 增加一个特征，并同步调整 `w`；
2. 构造两个输出的 `W` 和 `b`，手算第一个样本；
3. 解释为什么 `X.T @ error` 的形状能与 `w` 一致；
4. 故意使用错误长度的 `w`，说明应在矩阵乘法前怎样给出更清楚的错误。
