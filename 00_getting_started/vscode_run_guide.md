# VS Code 运行指南

## 从仓库根目录运行

在 VS Code 中打开整个仓库文件夹，而不是只打开某个 `.py` 文件。新建终端后运行：

```powershell
Get-Location
python 02_machine_learning/01_linear_regression/demo.py
```

`Get-Location` 应显示本仓库根目录。命令中的相对路径会从当前工作目录开始解释；如果当前目录错误，即使文件真实存在也会收到“找不到文件”。

## 避免工作目录混乱

- 从资源管理器或 VS Code 的仓库根目录打开终端；
- 不依赖编辑器临时生成的运行命令；
- 读写数据时先打印 `Path.cwd()`，再核对相对路径指向哪里；
- 专题 README 中的命令默认都从仓库根目录运行。
