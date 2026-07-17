"""模拟卷1第1题：不要查看参考答案，独立完成。"""

import argparse
from pathlib import Path


def run(input_path: Path, output_path: Path) -> None:
    raise NotImplementedError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
