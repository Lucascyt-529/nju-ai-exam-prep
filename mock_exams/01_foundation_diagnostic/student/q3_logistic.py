"""模拟卷1第3题：不要查看参考答案，独立完成。"""

import argparse
from pathlib import Path


def run(train_path: Path, validation_path: Path, test_path: Path,
        model_path: Path, predictions_path: Path, metrics_path: Path) -> None:
    raise NotImplementedError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    run(args.train, args.validation, args.test, args.model, args.predictions, args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
