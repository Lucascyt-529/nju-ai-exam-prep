"""运行前预测 Series、DataFrame 和单列选择的类型与形状。"""

import pandas as pd


def main() -> None:
    scores = pd.Series(
        [78.0, 91.0, 85.0], index=["小林", "小周", "小陈"], name="score"
    )
    students = pd.DataFrame(
        {
            "name": ["小林", "小周", "小陈"],
            "study_hours": [2.5, 4.0, 3.0],
            "score": [78.0, 91.0, 85.0],
        }
    )

    print("scores:")
    print(scores)
    print("scores shape:", scores.shape, "dtype:", scores.dtype)
    print("students:")
    print(students)
    print("shape:", students.shape)
    print("dtypes:")
    print(students.dtypes)
    print("one bracket:", type(students["score"]), students["score"].shape)
    print("two brackets:", type(students[["score"]]), students[["score"]].shape)


if __name__ == "__main__":
    main()
