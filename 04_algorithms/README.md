# 算法题

算法题模块训练完整程序，而不是只补 LeetCode 风格函数。每道题都要处理输入、输出、自测、复杂度和确定性规则，并能迁移到改变数据格式或输出要求的变式。

## 核心模式

### 数组与字符串

- [记录排序](../algorithms/01_sorting_records/README.md)
- [二分边界](../algorithms/02_binary_search_boundaries/README.md)
- [哈希与序列](../algorithms/03_hashing_sequences/README.md)
- [双指针与滑动窗口](../algorithms/10_two_pointers_sliding_window/README.md)
- [前缀和](../algorithms/11_prefix_sum/README.md)

### 栈、队列与堆

- [单调栈](../algorithms/12_stack_monotonic_stack/README.md)
- [堆与 Top K](../algorithms/13_heap_topk/README.md)

### 链表与树

- [反转链表](../algorithms/14_linked_list/README.md)
- [二叉树层序遍历](../algorithms/15_binary_tree/README.md)

### 搜索与回溯

- [网格最短路](../algorithms/05_grid_shortest_path/README.md)
- [生成所有子集](../algorithms/16_backtracking/README.md)

### 图

- [图遍历](../algorithms/04_graph_traversal/README.md)
- [Dijkstra](../algorithms/06_dijkstra_shortest_path/README.md)
- [拓扑排序](../algorithms/18_topological_sort/README.md)

### 动态规划

- [最少硬币](../algorithms/07_min_coin_change/README.md)
- [最长公共子序列](../algorithms/08_longest_common_subsequence/README.md)
- [0/1 背包](../algorithms/09_zero_one_knapsack/README.md)

### 贪心

- [区间合并](../algorithms/17_greedy_intervals/README.md)

## 每题固定要求

1. 先写清输入规模、数据结构和输出规则；
2. 给出朴素方案及其瓶颈，再选择算法；
3. 从空文件写出可直接运行的程序；
4. 覆盖空输入、重复值、不可达、平局等题目相关边界；
5. 说明时间和空间复杂度；
6. 至少完成一个变式，例如从标准输入改为文件输入，或增加路径/方案恢复。

仓库有材料不等于学习者已掌握。正式学习时按当前计划推进；最终验收不是填完
`starter.py`，而是能从空文件写出读取、算法、输出、自测和 `main()`，并把代表性错误记录到 [错题本](../records/bug_book/README.md)。
