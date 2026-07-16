"""运行前预测pivot后的形状、索引和列。"""

from pathlib import Path

import pandas as pd


def main() -> None:
    source = Path(__file__).resolve().parent / "data" / "long_scores.csv"
    long_frame = pd.read_csv(source)
    wide = long_frame.pivot(index="student_id", columns="course", values="score")
    print("long shape:", long_frame.shape)
    print("pivot shape before reset:", wide.shape)
    print("pivot index:", wide.index.tolist())
    print("pivot columns:", wide.columns.tolist())
    print("pivot columns.name:", wide.columns.name)
    reset = wide.reset_index()
    reset.columns.name = None
    print("after reset shape:", reset.shape)
    print(reset)


if __name__ == "__main__":
    main()
