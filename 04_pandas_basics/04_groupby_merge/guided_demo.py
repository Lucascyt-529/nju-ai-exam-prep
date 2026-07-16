"""运行前预测每个课程的输入行数和不同学生数。"""

from pathlib import Path

import pandas as pd


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    scores = pd.read_csv(data_dir / "scores.csv")
    grouped = scores.groupby("course").agg(
        record_count=("score", "size"),
        student_count=("student_id", "nunique"),
        mean_score=("score", "mean"),
    )
    print("input shape:", scores.shape)
    print("grouped index:", grouped.index.tolist())
    print("grouped columns:", grouped.columns.tolist())
    print(grouped)
    print("after reset_index shape:", grouped.reset_index().shape)


if __name__ == "__main__":
    main()
