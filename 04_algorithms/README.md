# 算法题

算法题模块训练完整程序，而不是只补 LeetCode 风格函数。每道题都要处理输入、输出、自测、复杂度和确定性规则，并能迁移到改变数据格式或输出要求的变式。

## 学习顺序

| 顺序 | 专题 | 核心模式 | 主要复杂度 |
| --- | --- | --- | --- |
| 1 | [记录排序](../algorithms/01_sorting_records/README.md) | 稳定归并排序、严格解析 | `O(n log n)` |
| 2 | [二分边界](../algorithms/02_binary_search_boundaries/README.md) | lower/upper bound、重复值 | `O(log n)` |
| 3 | [哈希与序列](../algorithms/03_hashing_sequences/README.md) | 频次、去重、两数配对 | 平均 `O(n)` |
| 4 | [图遍历](../algorithms/04_graph_traversal/README.md) | BFS、DFS、连通分量 | `O(V+E)` |
| 5 | [网格最短路](../algorithms/05_grid_shortest_path/README.md) | BFS、父节点、路径恢复 | `O(rows*cols)` |
| 6 | [Dijkstra](../algorithms/06_dijkstra_shortest_path/README.md) | 最小堆、陈旧条目、路径恢复 | `O((V+E) log V)` |
| 7 | [最少硬币](../algorithms/07_min_coin_change/README.md) | 一维 DP、方案恢复 | `O(amount*coins)` |
| 8 | [最长公共子序列](../algorithms/08_longest_common_subsequence/README.md) | 二维 DP、确定性恢复 | `O(nm)` |
| 9 | [0/1 背包](../algorithms/09_zero_one_knapsack/README.md) | 状态转移、下标恢复 | `O(n*capacity)` |

## 每题固定要求

1. 先写清输入规模、数据结构和输出规则；
2. 给出朴素方案及其瓶颈，再选择算法；
3. 从空文件写出可直接运行的程序；
4. 覆盖空输入、重复值、不可达、平局等题目相关边界；
5. 说明时间和空间复杂度；
6. 至少完成一个变式，例如从标准输入改为文件输入，或增加路径/方案恢复。

这些专题已有参考闭环，但尚未计入个人掌握。正式学习时按顺序推进，并把代表性错误记录到 [错题本](../records/bug_book/README.md)。
