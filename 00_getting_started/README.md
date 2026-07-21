# 开始使用

先确认终端当前目录是仓库根目录，也就是能看到 `README.md` 和 `pytest.ini` 的目录。

```powershell
Get-Location
python 02_machine_learning/01_linear_regression/demo.py
```

- [VS Code 运行与工作目录](vscode_run_guide.md)
- [自己定位错误](self_debugging.md)

三个运行入口用途不同：

| 入口 | 用途 |
| --- | --- |
| `demo.py` | 查看一组输入、期望输出和实际输出 |
| `check.py` | 用几组明确的期望值核对 `starter.py` |
| `python -m pytest ...` | 运行严格参考测试和仓库回归测试 |
