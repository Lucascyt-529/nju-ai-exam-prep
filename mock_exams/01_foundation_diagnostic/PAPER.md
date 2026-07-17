# 南京大学AI预推免机试训练：模拟卷1

总时长120分钟，总分150分。允许Python标准库与NumPy；不得使用pandas、sklearn或其他机器学习库。所有程序都必须能从命令行独立运行，错误信息写入标准错误并以非0状态退出。

## 第1题：缺失CSV统计（30分，建议20分钟）

输入CSV表头固定为`sample_id,feature_1,feature_2,feature_3`。特征空白表示缺失；`sample_id`不能为空或重复；显式`nan`/`inf`非法。

请输出：

```text
feature,mean,std,missing_count
feature_1,...,...,...
feature_2,...,...,...
feature_3,...,...,...
```

均值忽略缺失值。先用该列均值填补缺失值，再计算总体标准差（`ddof=0`）。均值和标准差保留6位小数。整列缺失应报错。

命令：

```text
python q1_csv_stats.py --input data/q1_features.csv --output q1_summary.csv
```

## 第2题：网格最短路（45分，建议30分钟）

输入依次为行列数、网格、起点和终点。`.`可通行，`#`不可通行；每步只能上下左右移动且代价为1。坐标从0开始。

方向扩展顺序固定为上、左、右、下。可达时输出最短距离和一条由该顺序确定的路径：

```text
distance=7
path=(0,0)->(0,1)->...
```

不可达时输出：

```text
distance=-1
path=NONE
```

命令：

```text
python q2_grid_bfs.py --input data/q2_grid.txt --output q2_path.txt
```

## 第3题：NumPy逻辑回归完整任务（75分，建议70分钟）

训练集和验证集表头为`sample_id,feature_1,feature_2,feature_3,label`，测试集没有`label`。标签只能为0或1，特征必须是有限数值。

要求：

1. 只用训练集均值和总体标准差标准化三份数据，零标准差替换为1；
2. 用NumPy实现稳定sigmoid和带L2正则的二分类逻辑回归；
3. 截距不参与L2；默认学习率0.2、迭代1500步、L2系数0.02；
4. 在验证集输出`accuracy`和`f1`，阈值为0.5；
5. 保存训练均值、安全标准差、权重和截距到`.npz`；
6. 重新加载模型后预测测试集；
7. 输出`sample_id,probability,label`，概率保留6位小数。

命令：

```text
python q3_logistic.py --train data/q3_train.csv --validation data/q3_validation.csv --test data/q3_test.csv --model q3_model.npz --predictions q3_predictions.csv --metrics q3_metrics.txt
```

指标文件固定为：

```text
accuracy=...
f1=...
```

## 通用自检要求

- 输出目录不存在时自动创建；
- 单行CSV仍保持二维特征形状；
- 不修改输入数组；
- 训练、验证、测试样本顺序不改变；
- 所有概率、损失、参数和输出必须有限；
- 测试集不得参与均值、标准差、参数或阈值选择。
