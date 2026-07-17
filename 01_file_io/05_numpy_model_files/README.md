# 文件读写5：NumPy数组与模型参数文件

## 为什么要单独学习

训练完成后只保存`weights`通常不够。若训练前使用了均值填补或标准化，预测新数据时还必须恢复训练阶段的`mean`和`scale`。否则代码能运行，结果却不是同一个模型。

## `.npy`与`.npz`

| 格式 | 内容 | 读取方式 | 适合场景 |
| --- | --- | --- | --- |
| `.npy` | 一个数组 | `np.load` | 单个矩阵或向量 |
| `.npz` | 多个带名字数组 | `np.load`后按键取值 | 模型参数包 |

本专题统一使用`allow_pickle=False`，只保存数值数组。打开目标文件为二进制流后再调用`np.save/np.savez`，可避免NumPy擅自为`model.data`追加`.npy/.npz`扩展名。

## 模型包形状

```text
mean.shape    == (n_features,)
scale.shape   == (n_features,)
weights.shape == (n_features,)
bias          == 数值标量
X.shape       == (n_samples, n_features)
```

预测时先计算`(X-mean)/scale`，再做矩阵向量乘法。必须验证键集合、数组形状、有限性和`scale>0`，不能因为文件能打开就相信内容正确。

## 学习顺序

1. 先用`.npy`保存并恢复一个二维数组；
2. 预测实际文件名，确认没有自动追加扩展名；
3. 保存四个模型参数并打印每个形状；
4. 关闭文件后重新加载，而不是继续使用内存里的原对象；
5. 对同一批`X`比较保存前后预测；
6. 删除一个键、改变一个形状，确认加载器明确拒绝。

运行引导演示：

```text
python 01_file_io/05_numpy_model_files/guided_demo.py
```

## 迁移题

1. 增加`threshold`标量并同步修改严格键集合；
2. 将二分类权重扩展为`(n_features,n_classes)`，写出预测形状；
3. 构造`scale`含0、权重含NaN和bias形状为`(1,)`的损坏文件；
4. 把训练和预测拆成两个独立进程，证明预测进程没有依赖训练内存。
