"""展示原始字符串、数值转换、去重和训练统计量填补。"""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    path = Path(__file__).resolve().parent / "data" / "dirty_students.csv"
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    print("raw dtypes:")
    print(raw.dtypes)
    print(raw)

    converted = raw.copy()
    converted["student_id"] = converted["student_id"].str.strip()
    converted["name"] = converted["name"].str.strip()
    for column in ["study_hours", "attendance"]:
        converted[column] = pd.to_numeric(
            converted[column].str.strip(), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    deduplicated = converted.drop_duplicates("student_id", keep="last").reset_index(
        drop=True
    )
    medians = deduplicated[["study_hours", "attendance"]].median()
    cleaned = deduplicated.copy()
    cleaned[["study_hours", "attendance"]] = cleaned[
        ["study_hours", "attendance"]
    ].fillna(medians)

    print("converted and deduplicated:")
    print(deduplicated)
    print("training medians:")
    print(medians)
    print("cleaned:")
    print(cleaned)


if __name__ == "__main__":
    main()
