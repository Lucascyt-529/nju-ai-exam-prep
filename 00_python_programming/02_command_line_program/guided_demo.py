"""引导演示：导入、main返回值和进程退出码是三个不同层次。"""

import importlib.util
from pathlib import Path
import tempfile


SOLUTION = Path(__file__).resolve().parent / "reference" / "solution.py"
DATA = Path(__file__).resolve().parent / "data" / "values.txt"


def main() -> None:
    spec = importlib.util.spec_from_file_location("cli_demo", SOLUTION)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载参考实现")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    print("导入模块时没有自动执行命令行程序")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "nested" / "report.txt"
        code = module.main(["--input", str(DATA), "--output", str(output), "--scale", "2"])
        print("main返回值:", code)
        print("输出已创建:", output.is_file())
        print(output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
