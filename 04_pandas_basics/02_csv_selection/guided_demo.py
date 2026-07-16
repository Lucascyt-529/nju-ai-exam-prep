"""运行前预测 CSV 读取、iloc 和 loc 的结果。"""

from pathlib import Path

import pandas as pd


def main() -> None:
    data_path = Path(__file__).resolve().parent / "data" / "students.csv"
    frame = pd.read_csv(data_path, dtype={"student_id": str})

    print(frame)
    print("shape:", frame.shape)
    print("columns:", frame.columns.tolist())
    print("dtypes:")
    print(frame.dtypes)
    print("positions 1:3:")
    print(frame.iloc[1:3])
    print("score >= 85:")
    print(frame.loc[frame["score"] >= 85, ["name", "score"]])


if __name__ == "__main__":
    main()
