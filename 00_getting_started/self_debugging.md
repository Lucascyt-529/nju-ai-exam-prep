# 自己定位错误

1. 先读 traceback 最后一行，确认异常类型和直接原因；
2. 再向上找第一处属于本仓库的文件名和行号；
3. 在该行前打印变量的值、`type(...)`，NumPy 数组还要打印 `.shape`；
4. 把数据缩小到 2 至 5 个可手算样本；
5. 分别检查数值、shape 和输入是否被原地修改；
6. 修复后先运行专题 `check.py`，再按需要运行 pytest。

示例：

```python
print("X:", X)
print("X type:", type(X))
print("X shape:", X.shape)
```

`demo.py` 用于查看输入输出样例，`check.py` 用于核对几个核心结果，pytest 用于严格回归验证。学习时只需要先使用前两个。
