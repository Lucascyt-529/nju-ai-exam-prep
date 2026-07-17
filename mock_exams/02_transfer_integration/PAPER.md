# 南京大学AI预推免机试训练：模拟卷2

总时长120分钟，总分150分。允许Python标准库与NumPy；不得使用pandas、sklearn、SciPy或其他机器学习/图算法库。所有程序必须从命令行独立运行，错误写入标准错误并以非0状态退出。

## 第1题：交易CSV分组汇总（25分，建议15分钟）

输入表头固定为`record_id,category,amount`。ID和类别不能为空，ID不能重复；金额必须是非负有限数值。

按类别输出记录数、总额和均值，先按总额降序，相同总额按类别字符串升序：

```text
category,count,total,mean
...
```

总额和均值固定保留6位小数，输出目录不存在时自动创建。

```text
python q1_group_summary.py --input data/q1_transactions.csv --output q1_summary.csv
```

## 第2题：带权图最短路（45分，建议30分钟）

输入第一行为结点数`n`和边数`m`，接着`m`行是`起点 终点 权重`，最后一行是查询起点和终点。图是无向图；结点名是无空白字符串；边权必须为非负有限数值；不得有自环、重复无向边或未出现在边中的孤立查询结点。

使用Dijkstra算法。多条最短路径距离相同时，输出结点序列按字符串字典序最小的完整路径：

```text
distance=13.000000
path=A->C->...
```

不可达：

```text
distance=INF
path=NONE
```

```text
python q2_dijkstra.py --input data/q2_graph.txt --output q2_path.txt
```

## 第3题：NumPy PCA+kNN完整任务（80分，建议75分钟）

训练/验证CSV表头为`sample_id,feature_1,feature_2,feature_3,label`，测试集无`label`。空白特征表示缺失；显式`nan/inf`非法；标签必须是整数，但不保证从0连续编号。

要求：

1. 候选`k`固定为`1,3,5`；每个候选只用训练集拟合预处理、PCA和kNN，在验证集按accuracy选择；同分选较小`k`；
2. 缺失值使用当前拟合数据各列均值填补；整列缺失报错；
3. 用当前拟合数据的总体均值/标准差标准化，零标准差替换为1；
4. 只用NumPy实现PCA：协方差为`X.T@X/n`，取最大两个特征值对应方向；每个分量绝对值最大的载荷固定为正；
5. kNN使用投影空间平方欧氏距离。邻居距离相同按训练行顺序；类别票数相同，选这些近邻中该类距离和较小者；仍相同选数值标签较小者；
6. 选定`k`后，合并训练集和验证集，从头拟合缺失值、标准化和PCA，得到最终kNN模型；测试集不得参与任何拟合或选参；
7. 保存并重新加载`mean,std,components,train_projection,train_labels,selected_k`到`.npz`，必须`allow_pickle=False`安全恢复；
8. 输出验证选参记录和测试预测。

命令：

```text
python q3_pca_knn.py --train data/q3_train.csv --validation data/q3_validation.csv --test data/q3_test.csv --model q3_model.npz --predictions q3_predictions.csv --metrics q3_metrics.txt
```

指标文件：

```text
selected_k=...
validation_accuracy=...
```

预测文件：

```text
sample_id,label
...
```

accuracy保留6位小数。预测保持测试行顺序。

## 通用自检

- 单行CSV保持`X:(1,d)`，标签保持`y:(n,)`；
- 所有学习到的统计量、特征值、投影和距离必须有限；
- PCA分量形状为`(2,3)`，投影为`(n,2)`；
- 不修改输入数组；
- 改变测试集内容不能改变保存模型；
- 不得硬编码正式数据输出。
