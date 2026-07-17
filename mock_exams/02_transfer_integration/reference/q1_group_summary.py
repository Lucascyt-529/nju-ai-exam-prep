"""模拟卷2第1题参考实现：严格CSV分组汇总。"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


HEADER = ["record_id", "category", "amount"]


def load_rows(path: Path) -> list[tuple[str, str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != HEADER:
            raise ValueError("输入表头必须严格为record_id,category,amount")
        rows = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[name] is None for name in HEADER):
                raise ValueError(f"第{line_number}行列数错误")
            record_id = row["record_id"].strip()
            category = row["category"].strip()
            if not record_id or not category:
                raise ValueError(f"第{line_number}行ID和类别不能为空")
            if record_id in seen:
                raise ValueError(f"record_id重复：{record_id}")
            seen.add(record_id)
            try:
                amount = float(row["amount"].strip())
            except ValueError as error:
                raise ValueError(f"第{line_number}行金额不是数值") from error
            if not np.isfinite(amount) or amount < 0:
                raise ValueError(f"第{line_number}行金额必须是非负有限数值")
            rows.append((record_id, category, amount))
    if not rows:
        raise ValueError("输入不能只有表头")
    return rows


def summarize(rows: list[tuple[str, str, float]]) -> list[tuple[str, int, float, float]]:
    groups: dict[str, list[float]] = {}
    for _, category, amount in rows:
        groups.setdefault(category, []).append(amount)
    result = []
    for category, amounts in groups.items():
        total = float(np.sum(np.asarray(amounts, dtype=float)))
        result.append((category, len(amounts), total, total / len(amounts)))
    return sorted(result, key=lambda item: (-item[2], item[0]))


def run(input_path: Path, output_path: Path) -> None:
    rows = summarize(load_rows(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["category", "count", "total", "mean"])
        for category, count, total, mean in rows:
            writer.writerow([category, count, f"{total:.6f}", f"{mean:.6f}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.input, args.output)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
