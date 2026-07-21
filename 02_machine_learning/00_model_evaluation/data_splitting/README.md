# 数据划分

训练集拟合参数，验证集选择算法、超参数、阈值和预处理方案，测试集只在方案确定后做最终评估。反复查看测试集再改方案，会让测试集泄露成验证集。

普通留出用固定随机种子打乱索引后切成三份。分层留出在每个类别内部划分后合并，使各份类别比例更稳定。固定 seed 用于复现实验。

```text
train_validation_test_split_indices(n_samples, validation_size, test_size, seed)
stratified_split_indices(y, validation_size, test_size, seed)
```

两者返回 train、validation、test 三个一维整数索引数组。先核对互不重叠、并集覆盖全部样本、相同 seed 结果相同；分层函数还要逐类划分。

训练均值、标准差、填充值等统计量只能从训练数据估计，再原样用于验证和测试。已有交叉验证由[进阶评估](../../advanced/README.md)导航，本次不复制高级框架。

```powershell
python 02_machine_learning/00_model_evaluation/data_splitting/demo.py
python 02_machine_learning/00_model_evaluation/data_splitting/check.py
```

常见错误：先标准化全数据再切分、索引重叠、未固定种子、少数类从验证集消失、用测试集挑超参数。自学时先实现普通留出，再做两类各4个样本的分层留出，最后解释三份数据的职责。
