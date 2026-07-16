"""运行前预测单列Series与单列DataFrame转换后的形状。"""

import numpy as np
import pandas as pd


def main() -> None:
    frame = pd.DataFrame({"x1": [1.0, 3.0, 5.0], "x2": [10.0, 20.0, 30.0], "label": [0, 1, 1]})
    X = frame.loc[:, ["x1", "x2"]].to_numpy(dtype=float)
    y_vector = frame.loc[:, "label"].to_numpy(dtype=int)
    y_column = frame.loc[:, ["label"]].to_numpy(dtype=int)
    print("X shape:", X.shape)
    print("Series label shape:", y_vector.shape)
    print("one-column DataFrame label shape:", y_column.shape)
    print("pandas std default:", frame[["x1", "x2"]].std().to_numpy())
    print("pandas std ddof=0:", frame[["x1", "x2"]].std(ddof=0).to_numpy())
    print("NumPy std default:", np.std(X, axis=0))


if __name__ == "__main__":
    main()
