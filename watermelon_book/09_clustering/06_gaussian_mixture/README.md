# 聚类 6：高斯混合聚类

高斯混合模型假设每个样本由 `k` 个高斯成分之一生成：

```text
p(x) = sum_k alpha_k * Normal(x | mean_k, covariance_k)
```

E步计算 `responsibilities.shape == (n_samples, n_components)`；M步用责任度软计数更新：

```text
N_k = sum_i responsibility[i,k]
alpha_k = N_k / n
mean_k = sum_i responsibility[i,k] * x_i / N_k
covariance_k = weighted outer products / N_k
```

参考实现支持完整协方差，使用Cholesky分解计算稳定对数密度，并给协方差对角加入很小正则，避免奇异矩阵。初始化使用固定随机种子的K-means++式中心选择和全局协方差。

GMM是软聚类；最终硬标签只是取最大责任度。与K-means相比，它可以学习不同大小、方向和密度的椭圆簇，但也会受初始化、局部最优和成分退化影响。

## 学生任务

1. 写出单个二维高斯的对数密度形状；
2. 在对数空间合并混合权重并归一化责任度；
3. 手算软计数和新均值；
4. 用外积更新完整协方差；
5. 检查权重和、责任度行和及协方差正定性；
6. 观察似然历史和标签交换。
