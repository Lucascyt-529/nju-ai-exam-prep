# Python完整程序2：命令行、main与错误码

## 完整程序的控制流

机试不是只补一个函数。一个可运行程序通常分成六层：

```text
load_values          只负责读取与格式检查
compute_statistics   只负责数值计算
save_report          只负责输出格式
run_pipeline         串联三步，不解析命令行
build_parser         只声明参数
main                 解析参数、处理预期错误、返回错误码
```

最后的入口保护：

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

它保证“直接运行文件”会启动程序，而“测试中导入模块”不会擅自读文件或退出进程。

## 为什么不用一个巨大main

拆成小函数能分别测试输入、算法和输出。迁移题改变文件格式时，也不会同时破坏数值计算。目的不是工程炫技，而是让无OJ自测更快。

## 错误边界

- 参数本身缺失：由`argparse`打印帮助并退出2；
- 文件不存在、内容非法：`main`捕获`OSError/ValueError`，写标准错误并返回2；
- 编程错误：不要用`except Exception`吞掉，应保留堆栈便于定位；
- 正常完成：返回0。

## 学习步骤

1. 先完成三个纯函数，不写命令行；
2. 直接用Python调用`run_pipeline`；
3. 再完成`build_parser`和`main(argv)`；
4. 用显式参数列表测试`main`，避免修改全局`sys.argv`；
5. 最后从终端启动新进程，检查真实退出码和标准错误。

运行演示：

```text
python 00_python_programming/02_command_line_program/guided_demo.py
```

正式样例：

```text
python 00_python_programming/02_command_line_program/reference/solution.py --input 00_python_programming/02_command_line_program/data/values.txt --output report.txt --scale 2
```

## 迁移题

1. 增加`--ddof`并只允许0或1；
2. 增加`--delimiter`，把每行一个数改成一行多个数；
3. 输入含非法值时输出行号，并验证进程退出码为2；
4. 将计算函数替换成标准化，保持命令行层不变；
5. 从另一个Python文件导入本模块，证明没有产生输出文件。
